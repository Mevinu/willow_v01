import argparse
import gc
import json
import os
import platform
import socket
import struct
import psutil
import requests
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from zeroconf import ServiceInfo, Zeroconf

SERVICE_TYPE = "_meshllm._tcp.local."


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
            raise ConnectionError("Upstream peer closed the connection.")
        pos += n
    return bytes(buf)


def recv_tensor(sock: socket.socket, device: torch.device):
    header = recv_exact(sock, 16)
    seq_pos, b, s, h = struct.unpack("!IIII", header)
    raw_bytes = recv_exact(sock, b * s * h * 2)
    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float16).reshape(b, s, h)
    return tensor.to(device), seq_pos


# =====================================================================
# 2. HTTP SHARD CACHE & LOADER
# =====================================================================
def ensure_local_shard(cfg: dict) -> str:
    os.makedirs("./.mesh_cache", exist_ok=True)
    start_l = cfg["start_layer"]
    end_l = cfg["end_layer"]
    shard_path = os.path.abspath(f"./.mesh_cache/{cfg['model_name']}_L{start_l}_{end_l}.gguf")

    if os.path.exists(shard_path):
        print(f"[+] Found cached shard on disk: {shard_path} (Instant boot!)")
        return shard_path

    keep_head = "1" if cfg["is_last_stage"] else "0"
    url = (
        f"http://{cfg['master_ip']}:{cfg['http_port']}/shard"
        f"?start={start_l}&end={end_l}&embd=1&head={keep_head}"
    )
    print(f"[*] Streaming layers {start_l}..{end_l - 1} from Coordinator ({url})...")

    # Download to a temp file and rename on success so an interrupted transfer never leaves a truncated shard in the cache
    tmp_path = shard_path + ".part"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            mb = 0.0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        mb += len(chunk) / (1024 * 1024)
                        print(f"\r    Downloaded: {mb:.1f} MB", end="", flush=True)
        os.replace(tmp_path, shard_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\n[+] Shard cached to {shard_path}")
    return shard_path


def load_worker_model(shard_path: str, is_last_stage: bool, device: torch.device):
    folder = os.path.dirname(shard_path)
    filename = os.path.basename(shard_path)

    print(f"[*] Loading shard '{filename}' onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(folder, gguf_file=filename, torch_dtype=torch.float16)

    # Workers never need input token embeddings (they receive float hidden states)
    model.model.embed_tokens = None

    # Middle workers don't need the final norm or LM head
    if not is_last_stage:
        model.model.norm = None
        model.lm_head = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(device)
    model.eval()
    return model


def run_layers(model, hidden_states, position_ids, past_key_values):
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    seq_len = hidden_states.shape[1]
    causal_mask = None
    if seq_len > 1:
        causal_mask = torch.full(
            (seq_len, seq_len), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    for layer in model.model.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        # transformers >= 4.54 returns a Tensor; older versions return a tuple
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return hidden_states


# =====================================================================
# 3. ZERO-CONFIG WORKER DAEMON
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Zero-Config Mesh-LLM Worker Node")
    parser.add_argument("--name", default=platform.node(), help="Node name on the network")
    parser.add_argument("--ctrl-port", type=int, default=10000, help="Control port for Coordinator handshake")
    parser.add_argument("--data-port", type=int, default=10001, help="Data port for incoming activation tensors")
    args = parser.parse_args()

    local_ip = get_local_ip()
    device = get_device()
    free_ram_mb = int(psutil.virtual_memory().available / (1024 * 1024))
    node_id = f"{args.name}-{args.ctrl_port}"

    # 1. Advertise this Worker on the LAN via mDNS (Bonjour / Zeroconf)
    zc = Zeroconf()
    service_info = ServiceInfo(
        SERVICE_TYPE,
        f"{node_id}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(local_ip)],
        port=args.ctrl_port,
        properties={
            "node_id": node_id,
            "ctrl_port": str(args.ctrl_port),
            "data_port": str(args.data_port),
            "free_ram_mb": str(free_ram_mb),
            "backend": device.type,
        },
    )
    zc.register_service(service_info)

    ctrl_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_server.bind(("0.0.0.0", args.ctrl_port))
    ctrl_server.listen(1)

    print(f"============================================================")
    print(f" Mesh-LLM Worker Active | ID: {node_id}")
    print(f" IP: {local_ip} | Backend: {device.type} | Free RAM: {free_ram_mb} MB")
    print(f" Broadcasting via mDNS... Waiting for Coordinator...")
    print(f"============================================================")

    try:
        while True:
            # 2. Wait for Coordinator Assignment
            ctrl_conn, addr = ctrl_server.accept()
            raw_cfg = ctrl_conn.recv(4096).decode("utf-8")
            cfg = json.loads(raw_cfg)
            print(f"\n[+] Received assignment from Coordinator at {addr[0]}:")
            print(f"    Model: {cfg['model_name']} | Assigned Layers: {cfg['start_layer']}..{cfg['end_layer'] - 1}")

            # 3. Stream ONLY our assigned GGUF layers over HTTP & load onto GPU/CPU
            shard_path = ensure_local_shard(cfg)
            model = load_worker_model(shard_path, cfg["is_last_stage"], device)

            # 4. Open incoming data socket BEFORE signaling READY so upstream can connect immediately
            data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            data_server.bind(("0.0.0.0", args.data_port))
            data_server.listen(1)

            ctrl_conn.sendall(b"READY")
            ctrl_conn.close()

            # 5. Connect upstream (previous stage) and downstream (next stage or Coordinator)
            upstream_conn, _ = data_server.accept()
            upstream_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            downstream_sock.connect((cfg["next_ip"], cfg["next_port"]))
            print(f"[+] Ring connected! Upstream -> [Layers {cfg['start_layer']}..{cfg['end_layer']-1}] -> {cfg['next_ip']}:{cfg['next_port']}")

            # 6. Execute Pipeline Forward Passes
            kv_cache = DynamicCache()
            with torch.no_grad():
                while True:
                    try:
                        hidden_states, seq_pos = recv_tensor(upstream_conn, device)
                    except ConnectionError:
                        print("[*] Session complete. Returning to idle standby...\n")
                        break

                    seq_len = hidden_states.shape[1]
                    if seq_pos == 0:
                        kv_cache = DynamicCache()
                        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
                    else:
                        position_ids = torch.tensor([[seq_pos]], device=device)

                    hidden_states = run_layers(model, hidden_states, position_ids, kv_cache)

                    if cfg["is_last_stage"]:
                        # Final Worker computes Norm + LM Head and sends 4-byte token ID back to Coordinator
                        hidden_states = model.model.norm(hidden_states)
                        logits = model.lm_head(hidden_states[:, -1:, :])
                        next_token_id = int(torch.argmax(logits, dim=-1).item())
                        downstream_sock.sendall(struct.pack("!I", next_token_id))
                    else:
                        # Middle Worker forwards the 8KB hidden-state tensor to the next Worker in the ring
                        send_tensor(downstream_sock, hidden_states, seq_pos)

            upstream_conn.close()
            downstream_sock.close()
            data_server.close()

    except KeyboardInterrupt:
        print("\n[*] Shutting down worker...")
    finally:
        zc.unregister_service(service_info)
        zc.close()
        ctrl_server.close()


if __name__ == "__main__":
    main()