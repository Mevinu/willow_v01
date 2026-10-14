# Willow v0.1: Frictionless Heterogeneous Pipeline Parallelism for LLMs

Willow is a zero-config, cross-platform distributed inference tool that lets you run massive Large Language Models (LLMs) across multiple heterogeneous devices (macOS Apple Silicon, NVIDIA CUDA, Linux/Windows PCs, and CPUs) on your local network. 

Instead of requiring complex manual network configurations or pushing gigabytes of weights over Wi-Fi on every boot, Willow automatically discovers peers via mDNS, calculates dynamic layer distributions, and streams memory-mapped GGUF shards on-the-fly.

---

## Architecture & How It Works

Willow implements a **True Peer-to-Peer Ring Pipeline** combined with **OS Memory Mapping (`mmap`)**:

1. **Zero-Config mDNS Discovery:** Workers broadcast their hardware capabilities (RAM, backend type like `metal` or `cuda`) over the local network using Zeroconf/Bonjour. No IP addresses or ports need to be typed manually.
2. **Dynamic Layer Balancing:** The Coordinator analyzes all connected nodes and calculates an optimal layer allocation matrix ($\text{Layer}_0 \rightarrow \text{Layer}_N$) proportional to each node's available memory and compute power.
3. **Zero-RAM GGUF Slicing Stream:** The Coordinator opens the monolithic `.gguf` model file on its SSD using `mmap` (taking ~10 MB of RAM). When a worker requests its assigned layers, the Coordinator dynamically synthesizes a valid mini-GGUF header and streams *only* the raw quantized tensor bytes straight from disk over HTTP.
4. **Local Caching & Instant Boots:** Workers cache their assigned shard slice locally in `./.mesh_cache/`. Subsequent cluster restarts boot in **under 1 second** without re-downloading.
5. **Low-Latency Tensor Ring:** During generation, nodes pass a tiny 8 KB hidden-state activation vector ($[1, 1, \text{hidden\_dim}]$ FP16) sequentially through a TCP socket ring (`Master -> Worker 1 -> Worker 2 -> Master`).

---

## Project Structure

```text
willow-v01/
├── coordinator.py     # mDNS browser, Shard HTTP streamer, layer partitioner, & Stage 0 head node
├── worker.py          # Zero-config mDNS advertiser, shard downloader, & pipeline worker daemon
└── requirements.txt   # Python dependency manifest
```

---

## Installation & Requirements

Ensure Python 3.10+ is installed on all machines in your local network.

Install the required dependencies on every node:

```bash
pip install torch transformers accelerate gguf>=0.10.0 zeroconf psutil requests protobuf
```

---

## Quickstart Guide

### Step 1: Start Worker Nodes (On any auxiliary laptops/PCs)
Workers require **zero arguments** and **do not need the `.gguf` file on their disk**. Simply run:

```bash
python worker.py
```
*The worker will automatically detect its hardware, broadcast its presence on the local network via mDNS, and wait for instructions.*

### Step 2: Start the Coordinator & Run Inference (On the host machine)
Place your GGUF model file (e.g., Llama-3.2 or Qwen-2.5) on the host machine and run:

```bash
python coordinator.py --gguf ./your-model.gguf --prompt "Explain distributed computing in two simple sentences:"
```

The coordinator will:
1. Scan the LAN for active workers via mDNS.
2. Display the computed pipeline allocation plan.
3. Stream the sliced layer shards to each worker.
4. Execute the ring pipeline and print tokens live to your terminal as they generate.

---

## Technical Specifications

- **Model Support:** Native compatibility with any GGUF quantized model (`Q4_K_M`, `Q8_0`, etc.).
- **Transport Layer:** Raw TCP sockets with `TCP_NODELAY` enabled and custom 16-byte binary tensor framing for microsecond activation transfers.
- **Memory Footprint:** Uses OS `mmap` page caches; both Coordinator and Worker RAM usage remain flat regardless of whether the model is 3 GB or 100 GB.

---

## Roadmap

- [ ] Add an OpenAI-compatible FastAPI streaming endpoint (`/v1/chat/completions`) for WebUI compatibility.
- [ ] Implement hot-swapping and auto-healing when a worker drops from the Wi-Fi mesh.
- [ ] Provide pre-compiled standalone binaries for macOS, Linux, and Windows.
