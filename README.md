# neural-life
Neural life simulation with GPU acceleration

#What's running:

80×110 grid of hybrid cells, each carrying a Brainfuck-style program tape + NCA hidden state channels (nca[0..3])
Cells execute instructions (the full extended set: + - > < [ ] . , $ X & 8 ? @ #) and consume energy — hit zero and they die
Replication triggers when a cell has enough energy and a free neighbor — it copies its tape with mutation into the target
NCA coupling blends each cell's hidden channels with its Moore neighborhood every step, creating coherent clusters
The . and , instructions let cells literally overwrite each other's tape memory — the core BFF interaction mechanic

Controls:

Radiation — scrambles a random zone, forcing repair and novelty
Food Pulse — energy injection into a region to spark growth
Speed ×1/3/8 — scale simulation throughput

Live diagnostics track Shannon entropy (dropping entropy = dominant replicator emerging), alive count, replicator fraction, and NCA cohesion — with automatic phase detection from Soup → Body → Brain → Voice.

#Here's the full WebGPU migration. Here's what changed architecturally:
GPU cell struct (80 bytes, WGSL CellState):

tape: array<u32,8> — 4 BF bytes packed per u32, offset 0
ip, dp, energy, age, alive — u32s at offsets 32–48
3 padding u32s to hit the 16-byte alignment boundary
nca: vec4<f32> — at offset 64, all 4 NCA channels

Ping-pong pattern: Two storage buffers A/B — both bind groups (bgAB, bgBA) are pre-built at init. Each compute pass swaps which is read vs. written; [readBuf, writeBuf] = [writeBuf, readBuf] after every dispatch. For speed×3 or ×8, each sub-step is its own CommandEncoder so uniforms (tick, seed) update between passes.
Async readback loop: After gpuStep(n), a copyBufferToBuffer command copies the latest read buffer into a MAP_READ readback buffer. mapAsync(READ) awaits the GPU fence, the mapped range is copied into a shared displayBuffer ArrayBuffer, then unmapped. draw() and updateStats() read from this same buffer — meaning both paths (CPU and GPU) use identical rendering code.
Event injection (Radiation, Food, Reset): Events modify displayBuffer directly with byte-level tape patching, then call device.queue.writeBuffer(readBuf, ...) to push the patch back into the GPU pipeline. One frame of latency, imperceptible.
CPU fallback is preserved exactly — if WebGPU is unavailable, cpuLoop() runs the full JS simulation, packs to displayBuffer each frame, and drives the same draw path. The HUD badge shows WebGPU ✓ or CPU mode accordingly.
