import argparse
import gc
import http.server
import json
import os
import socket
import socketserver
import struct
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse

import gguf
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

SERVICE_TYPE = "_meshllm._tcp.local."
ALIGNMENT = 32


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# 1. BINARY TENSOR PROTOCOL OVER TCP
# =====================================================================
def send_tensor(sock: socket.socket, tensor: torch.Tensor, seq_pos: int):
    cpu_tensor = tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
    raw_bytes = cpu_tensor.numpy().tobytes()
    b, s, h = cpu_tensor.shape
    header = struct.pack("!IIII", seq_pos, b, s, h)
    sock.sendall(header + raw_bytes)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < num_bytes:
        n = sock.recv_into(view[pos:])
        if n == 0:
            raise ConnectionError("Peer closed the connection.")
        pos += n
    return bytes(buf)


# =====================================================================
# 2. ZERO-RAM ON-THE-FLY GGUF SHARD STREAMER (HTTP SERVER)
# =====================================================================
def pad_size(size: int, alignment: int = ALIGNMENT) -> int:
    return (size + alignment - 1) // alignment * alignment


def copy_gguf_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, arch: str, num_sliced_layers: int):
    """Copies all header metadata & tokenizer arrays from the original GGUF to the virtual shard."""
    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}

    for field in reader.fields.values():
        if field.name in skip_keys:
            continue
        if field.name == f"{arch}.block_count":
            writer.add_uint32(field.name, num_sliced_layers)
            continue

        if not field.types:
            continue
        vtype = field.types[0]

        try:
            if vtype == gguf.GGUFValueType.ARRAY:
                sub_type = field.types[1]
                arr = []
                for idx in field.data:
                    if sub_type == gguf.GGUFValueType.STRING:
                        arr.append(str(bytes(field.parts[idx]), encoding="utf-8", errors="ignore"))
                    else:
                        arr.append(field.parts[idx][0])
                writer.add_array(field.name, arr)
            elif vtype == gguf.GGUFValueType.STRING:
                val = str(bytes(field.parts[field.data[0]]), encoding="utf-8", errors="ignore")
                writer.add_string(field.name, val)
            else:
                val = field.parts[field.data[0]][0]
                if vtype == gguf.GGUFValueType.UINT32:
                    writer.add_uint32(field.name, int(val))
                elif vtype == gguf.GGUFValueType.INT32:
                    writer.add_int32(field.name, int(val))
                elif vtype == gguf.GGUFValueType.FLOAT32:
                    writer.add_float32(field.name, float(val))
                elif vtype == gguf.GGUFValueType.BOOL:
                    writer.add_bool(field.name, bool(val))
                elif vtype == gguf.GGUFValueType.UINT64:
                    writer.add_uint64(field.name, int(val))
        except Exception:
            pass


def stream_virtual_gguf_shard(gguf_path: str, start_layer: int, end_layer: int, keep_embd: bool, keep_head: bool):
    """Yields a valid GGUF file containing ONLY [start_layer, end_layer) via OS mmap."""
    reader = gguf.GGUFReader(gguf_path, mode="r")
    arch_field = reader.get_field("general.architecture")
    arch = str(bytes(arch_field.parts[arch_field.data[0]]), encoding="utf-8")
    num_sliced_layers = end_layer - start_layer

    kept_tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            if start_layer <= layer_idx < end_layer:
                parts[1] = str(layer_idx - start_layer)  # Re-index layers to 0..N-1
                kept_tensors.append((".".join(parts), tensor))
        elif name == "token_embd.weight" and keep_embd:
            kept_tensors.append((name, tensor))
        elif name in ("output_norm.weight", "output.weight") and keep_head:
            kept_tensors.append((name, tensor))

    # Build temporary header in ~2MB disk file
    with tempfile.NamedTemporaryFile(delete=True) as tmp_header:
        writer = gguf.GGUFWriter(tmp_header.name, arch=arch)
        copy_gguf_metadata(reader, writer, arch, num_sliced_layers)

        for new_name, tensor in kept_tensors:
            writer.add_tensor_info(
                new_name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.data.nbytes,
                raw_dtype=tensor.tensor_type,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.flush()

        tmp_header.seek(0)
        yield tmp_header.read()

    # Stream raw quantized tensor bytes straight from SSD mmap in 4MB chunks
    CHUNK_SIZE = 4 * 1024 * 1024
    for _, tensor in kept_tensors:
        byte_view = memoryview(tensor.data).cast("B")
        nbytes = len(byte_view)
        for offset in range(0, nbytes, CHUNK_SIZE):
            yield bytes(byte_view[offset : offset + CHUNK_SIZE])

        padding_needed = pad_size(nbytes, ALIGNMENT) - nbytes
        if padding_needed > 0:
            yield b"\x00" * padding_needed


class ShardHTTPHandler(http.server.BaseHTTPRequestHandler):
    gguf_path = ""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/shard":
            self.send_error(404)
            return

        params = parse_qs(parsed.query)
        start_layer = int(params["start"][0])
        end_layer = int(params["end"][0])
        keep_embd = params.get("embd", ["1"])[0] == "1"
        keep_head = params.get("head", ["1"])[0] == "1"

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        for chunk in stream_virtual_gguf_shard(self.gguf_path, start_layer, end_layer, keep_embd, keep_head):
            self.wfile.write(chunk)

    def log_message(self, format, *args):
        return  # Keep console clean


def start_http_shard_server(gguf_path: str, port: int):
    ShardHTTPHandler.gguf_path = gguf_path
    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), ShardHTTPHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# =====================================================================
# 3. mDNS ZERO-CONFIG WORKER DISCOVERY
# =====================================================================
class WorkerDiscoveryListener(ServiceListener):
    def __init__(self):
        self.workers = {}

    def add_service(self, zc: Zeroconf, type_: str, name: str):
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])
            props = {k.decode(): v.decode() for k, v in info.properties.items()}
            worker_id = props.get("node_id", name)
            self.workers[worker_id] = {
                "node_id": worker_id,
                "ip": ip,
                "ctrl_port": int(props.get("ctrl_port", 10000)),
                "data_port": int(props.get("data_port", 10001)),
                "free_ram_mb": int(props.get("free_ram_mb", 4096)),
                "backend": props.get("backend", "cpu"),
            }
            print(
                f"  [+] Discovered Worker '{worker_id}' at {ip} "
                f"({self.workers[worker_id]['free_ram_mb']} MB free, {self.workers[worker_id]['backend']})"
            )

    def remove_service(self, zc: Zeroconf, type_: str, name: str):
        pass

    def update_service(self, zc: Zeroconf, type_: str, name: str):
        pass


# =====================================================================
# 4. DYNAMIC LAYER SOLVER & LOCAL EXECUTION
# =====================================================================
def get_gguf_layer_count(gguf_path: str) -> int:
    reader = gguf.GGUFReader(gguf_path, mode="r")
    arch_field = reader.get_field("general.architecture")
    arch = str(bytes(arch_field.parts[arch_field.data[0]]), encoding="utf-8")
    block_field = reader.get_field(f"{arch}.block_count")
    return int(block_field.parts[block_field.data[0]][0])


def partition_layers(total_layers: int, nodes: list) -> list:
    """Allocates layer slices across nodes proportionally to their free RAM & GPU weight."""
    scores = []
    for n in nodes:
        multiplier = 1.5 if n["backend"] in ("cuda", "mps") else 1.0
        scores.append(max(512, n["free_ram_mb"]) * multiplier)

    total_score = sum(scores)
    assignments = []
    current_layer = 0

    for i, n in enumerate(nodes):
        if i == len(nodes) - 1:
            end_layer = total_layers
        else:
            count = max(1, round(total_layers * (scores[i] / total_score)))
            end_layer = min(total_layers - (len(nodes) - 1 - i), current_layer + count)

        assignments.append({**n, "start_layer": current_layer, "end_layer": end_layer})
        current_layer = end_layer

    return assignments


def run_layers(model, hidden_states, position_ids, past_key_values):
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    seq_len = hidden_states.shape[1]
    causal_mask = None
    if seq_len > 1:
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float16).min, device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    for layer in model.model.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

    return hidden_states


def load_master_slice(gguf_path: str, end_layer: int, total_layers: int, device: torch.device):
    """Creates a local cached shard for Stage 0 so the Master also never loads the full 40GB file into RAM."""
    os.makedirs("./.mesh_cache", exist_ok=True)
    model_tag = os.path.basename(gguf_path).replace(".gguf", "")
    shard_path = os.path.abspath(f"./.mesh_cache/{model_tag}_master_0_{end_layer}.gguf")

    if not os.path.exists(shard_path):
        print(f"[*] Creating local Master shard (Layers 0..{end_layer - 1}) in ./.mesh_cache ...")
        keep_head = end_layer == total_layers
        with open(shard_path, "wb") as f:
            for chunk in stream_virtual_gguf_shard(gguf_path, 0, end_layer, keep_embd=True, keep_head=keep_head):
                f.write(chunk)

    folder = os.path.dirname(shard_path)
    filename = os.path.basename(shard_path)
    tokenizer = AutoTokenizer.from_pretrained(folder, gguf_file=filename)
    model = AutoModelForCausalLM.from_pretrained(folder, gguf_file=filename, torch_dtype=torch.float16)

    if end_layer < total_layers:
        model.model.norm = None
        model.lm_head = None

    gc.collect()
    model.to(device)
    model.eval()
    return model, tokenizer


# =====================================================================
# 5. MAIN COORDINATOR ORCHESTRATION LOOP
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Zero-Config Heterogeneous LLM Pipeline Coordinator")
    parser.add_argument("--gguf", required=True, help="Path to the .gguf model file")
    parser.add_argument("--http-port", type=int, default=8765, help="Port to stream GGUF shards from")
    parser.add_argument("--token-port", type=int, default=9998, help="Port to receive final token IDs")
    parser.add_argument("--wait-seconds", type=int, default=5, help="Seconds to scan for LAN workers via mDNS")
    parser.add_argument("--prompt", default="Explain how distributed systems work in 3 short bullet points:")
    args = parser.parse_args()

    local_ip = get_local_ip()
    device = get_device()
    total_layers = get_gguf_layer_count(args.gguf)
    model_name = os.path.basename(args.gguf).replace(".gguf", "")

    print(f"============================================================")
    print(f" Mesh-LLM Coordinator | IP: {local_ip} | Device: {device}")
    print(f" Model: {model_name} ({total_layers} layers)")
    print(f"============================================================")

    # 1. Start HTTP Zero-RAM GGUF Shard Server
    start_http_shard_server(args.gguf, args.http_port)
    print(f"[*] Started Zero-RAM GGUF Shard Streamer on http://{local_ip}:{args.http_port}/shard")

    # 2. Discover Workers on LAN via mDNS
    print(f"[*] Scanning local network for workers via mDNS ({args.wait_seconds}s)...")
    zc = Zeroconf()
    listener = WorkerDiscoveryListener()
    browser = ServiceBrowser(zc, SERVICE_TYPE, listener)
    time.sleep(args.wait_seconds)
    zc.close()

    # 3. Build Cluster Topology (Master is always Stage 0)
    master_free_mb = int(psutil.virtual_memory().available / (1024 * 1024))
    all_nodes = [
        {
            "node_id": "master",
            "ip": local_ip,
            "free_ram_mb": master_free_mb,
            "backend": device.type,
        }
    ] + list(listener.workers.values())

    plan = partition_layers(total_layers, all_nodes)
    print("\n[*] Computed Heterogeneous Pipeline Allocation Plan:")
    for idx, stage in enumerate(plan):
        print(
            f"    Stage {idx} ({stage['node_id']} @ {stage['ip']} [{stage['backend']}]): "
            f"Layers {stage['start_layer']} .. {stage['end_layer'] - 1}"
        )

    # 4. Configure Remote Workers (in reverse order so downstream sockets are listening first)
    token_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    token_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    token_listener.bind(("0.0.0.0", args.token_port))
    token_listener.listen(1)

    for idx in range(len(plan) - 1, 0, -1):
        worker = plan[idx]
        is_last = idx == len(plan) - 1
        next_ip = local_ip if is_last else plan[idx + 1]["ip"]
        next_port = args.token_port if is_last else plan[idx + 1]["data_port"]

        config_payload = {
            "model_name": model_name,
            "master_ip": local_ip,
            "http_port": args.http_port,
            "start_layer": worker["start_layer"],
            "end_layer": worker["end_layer"],
            "total_layers": total_layers,
            "is_last_stage": is_last,
            "next_ip": next_ip,
            "next_port": next_port,
        }

        print(f"\n[*] Initializing {worker['node_id']} ({worker['ip']}:{worker['ctrl_port']})...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ctrl_sock:
            ctrl_sock.connect((worker["ip"], worker["ctrl_port"]))
            ctrl_sock.sendall(json.dumps(config_payload).encode("utf-8"))
            resp = ctrl_sock.recv(64).decode("utf-8")
            if resp != "READY":
                raise RuntimeError(f"Worker {worker['node_id']} failed initialization: {resp}")
        print(f"[+] {worker['node_id']} downloaded shard & opened pipeline socket!")

    # 5. Load Master's Own Layer Slice (Stage 0)
    master_end_layer = plan[0]["end_layer"]
    model, tokenizer = load_master_slice(args.gguf, master_end_layer, total_layers, device)

    # Connect to Stage 1 Worker (if multi-node)
    downstream_sock = None
    tail_token_sock = None
    if len(plan) > 1:
        downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        downstream_sock.connect((plan[1]["ip"], plan[1]["data_port"]))

        tail_token_sock, _ = token_listener.accept()
        tail_token_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # 6. Run Distributed Autoregressive Generation Loop
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    kv_cache = DynamicCache()
    seq_pos = 0

    print(f"\n============================================================")
    print(f"Prompt: {args.prompt}\nResponse: ", end="", flush=True)
    t_start = time.time()
    generated_tokens = 0

    with torch.no_grad():
        for step in range(80):
            seq_len = input_ids.shape[1]
            if step == 0:
                position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
            else:
                position_ids = torch.tensor([[seq_pos]], device=device)

            hidden_states = model.model.embed_tokens(input_ids)
            hidden_states = run_layers(model, hidden_states, position_ids, kv_cache)

            if len(plan) == 1:
                # Single-node fallback if 0 workers joined
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states[:, -1:, :])
                next_token_id = int(torch.argmax(logits, dim=-1).item())
            else:
                # Send activation tensor to Stage 1 Worker & wait for final token ID from Tail Worker
                send_tensor(downstream_sock, hidden_states, seq_pos)
                raw_token = recv_exact(tail_token_sock, 4)
                next_token_id = struct.unpack("!I", raw_token)[0]

            seq_pos += seq_len
            generated_tokens += 1
            print(tokenizer.decode([next_token_id]), end="", flush=True)

            if next_token_id == tokenizer.eos_token_id:
                break

            input_ids = torch.tensor([[next_token_id]], device=device)

    elapsed = time.time() - t_start
    print(f"\n============================================================")
    print(f"[+] Generated {generated_tokens} tokens in {elapsed:.2f}s ({generated_tokens / elapsed:.2f} tok/s)")

    if downstream_sock:
        downstream_sock.close()
    if tail_token_sock:
        tail_token_sock.close()


if __name__ == "__main__":
    main()