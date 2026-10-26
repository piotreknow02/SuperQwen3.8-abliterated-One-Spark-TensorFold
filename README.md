<h1 align="center">SuperQwen3.8 Flash Next (abliterated) on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://github.com/piotreknow02">piotreknow02</a><br>
  based on the original DGX Spark recipe by <a href="https://github.com/MiaAI-Lab">Mia'a AI Lab</a> (<a href="https://x.com/MiaAI_lab">X</a>)</sub>
</p>

Serve **SuperQwen3.8 Flash Next (abliterated): NVFP4 (ModelOpt), with the MTP draft head** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with **5 concurrent requests at the full 262,144-token context** and **image and video input**. It runs [TensorFold](https://github.com/ashhart/TensorFold) v0.6.1 (`17c73e1`) in NVIDIA's PyTorch container, plus `patches/0002-flash-next-v061.patch` (video input and many images on TensorFold's Flash Next vision, copy drafts, SSD read-ahead, first token before the next draft).

- Checkpoint: [`Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark`](https://huggingface.co/Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark)
  (NVFP4 routed experts in blocks of 16, bf16 linears, FP8 n-gram tables, MTP draft head; the TensorFold CUDA engine reads NVFP4 or MLX 4-bit group-32 only, so the MLX group-64 packaging of this model cannot be served here)
- Base model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- API model id: `SuperQwen3.8-Flash-Next-abliterated`
- KV pool: **1,310,720 tokens** (5 streams x 262,144, int8 KV cache, ~23.4 GiB)
- Images and videos in chat messages (`image_url` / `video_url` parts)
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## What "abliterated" means

"Abliteration" removes the model's alignment/safety training. The base Qwen3.8-Flash-Next is a multimodal model with a vision tower and MTP draft head; this abliterated version was created by Jiunsong by removing the safety/alignment overrides. It retains the full vision tower, the MTP draft head and the 262,144-token context window. The recipe serves it with MTP drafting (`MTP_DRAFTS=6`), the same way the base-model recipe serves `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`.

> **Note:** TensorFold's CUDA kernels read 4-bit weights in groups of 32 or NVFP4 (ModelOpt) in blocks of 16. The earlier `SuperQwen3.8-Flash-Next-abliterated-MLX-4bit` packaging (group size 64, no MTP head) cannot be served on a DGX Spark; this recipe uses the NVFP4 DGX Spark variant of the same abliterated model.

## Performance

One DGX Spark, on TensorFold v0.6.1 (5 streams x 262,144, int8 KV cache, image input on), measured through the
OpenAI API with the earlier MLX 4-bit checkpoint of the same abliterated model, served with `--no-drafts` (that
packaging had no MTP head, and read its n-gram tables from SSD). The NVFP4 variant serves the same architecture
with MTP drafting and memory-mapped tables; re-measure after switching.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 63.6 tok/s | 63.6 tok/s | 66 ms |
| 2 | 85.6 tok/s | 43.8 tok/s | 226 ms |
| 4 | 114.5 tok/s | 29.7 tok/s | 304 ms |

**Decode, code**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 96.9 tok/s | 96.9 tok/s | 206 ms |
| 2 | 133.9 tok/s | 67.0 tok/s | 210 ms |
| 4 | 166.4 tok/s | 44.8 tok/s | 292 ms |

**Prefill**

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | --- | --- | --- |
| 8k | 8,228 | 2,421 tok/s | 3.40 s |
| 16k | 16,423 | 2,468 tok/s | 6.66 s |
| 32k | 32,805 | 2,462 tok/s | 13.33 s |
| 64k | 65,571 | 2,391 tok/s | 27.43 s |
| 128k | 131,110 | 2,180 tok/s | 60.13 s |

The abliterated version uses the same Qwen4Exp architecture as the base Qwen3.8-Flash-Next, so decode and prefill throughput are comparable to the Vontra MTP variant. `docs/v061.md` has the v0.6.0-to-v0.6.1 comparison.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU: the default setting needs ~101 GiB free when the server starts (see [KV pool and memory](#kv-pool-and-memory)).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~185 GB free disk on a fresh machine: ~150 GB for the checkpoint download under `~/.cache/huggingface` (~136 GB) and ~35 GB for the image under Docker's root (~24 GB); `scripts/prepare.sh` checks both.
- Optional: the `hf` CLI on the host (faster, resumable download) and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`.

## Quick start

```bash
git clone https://github.com/piotreknow02/SuperQwen3.8-abliterated-Single-DGX-Spark-TensorFold.git
cd SuperQwen3.8-abliterated-Single-DGX-Spark-TensorFold
./start.sh
```

That is all. The first run sets everything up (see below): it gets the image (pulling the ~11 GB prebuilt image when a matching tag is published, building it locally otherwise) and downloads the ~136 GiB checkpoint, then compiles the CUDA kernels for the GB10 (a few minutes, once). Later starts take a few minutes to load the weights. `start.sh` shows each step, the server's log and the loading progress, runs a smoke test, prints `SuperQwen3.8-Flash-Next-abliterated is now LIVE! on port 8080` with the endpoint, and returns you to the shell.

```bash
curl -s http://<spark-address>:8080/v1/models

curl -s http://<spark-address>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "SuperQwen3.8-Flash-Next-abliterated",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8080/v1"` and the model `SuperQwen3.8-Flash-Next-abliterated`.
Streaming, tool calls (typed parameters, e.g. arrays come back as JSON arrays), reasoning content, images and videos are supported. The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`;
a request without one gets `MAX_TOKENS` (32,768).

```bash
./start.sh restart                            # restart it, e.g. after changing a setting
./stop.sh                                     # stop the server and free the GPU memory
docker logs -f superqwen38-abliterated-tf     # server log
curl -s http://<spark-address>:8080/health    # busy flag and live token totals
```

## Images and video

The model's own vision tower (27 layers, 0.84 GiB, from the same checkpoint) turns images and video frames into
tokens, placed with Qwen's 3-D rotary positions, the way the reference implementation does. Send them as OpenAI-style
content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<spark-address>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "SuperQwen3.8-Flash-Next-abliterated",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`); with
OpenAI's Python client, pass the same dicts in `messages`.

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (all of a chat's turns count), 10 MB each, 64 MB in all | up to 2, 64 MB each, 96 MB in all, up to an hour of footage |
| Tokens | up to 16,384 for all images, at most 4,096 an image (50 images: ~320 each; `"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames, spread over the whole video), each pair of frames one timestamped block; up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image
and video prompts are not kept for prefix reuse, so each turn of a chat with images processes them again. A request
body can be up to 96 MiB (base64 makes data URLs a third larger than the files). Text requests are unaffected:
their replies stay byte-identical with vision on. `VISION=0 ./start.sh restart` serves text only.

## KV pool and memory

TensorFold gives every stream its own cache for a full window, so the KV pool is streams x window:

| | Default |
| --- | ---: |
| Streams (`PARALLEL`) | 5 |
| Window per stream (`CONTEXT`, the model's native maximum) | 262,144 tokens |
| **KV pool** | **1,310,720 tokens** (4 streams: 1,048,576) |
| KV precision (`KV_DTYPE`) | int8 (an fp16 scale per 32 values) |
| Memory a stream, allocated (server log) | 4,799 MiB: the KV cache, the sparse-attention index and the stream's own buffers |
| **Memory for the pool, allocated** | **~23.4 GiB** (5 x 4,799 MiB) |

The server reports these at every start: `5 streams of 262144 prompt/reply tokens (4799 MiB a stream)` and
`startup estimate 98.31 GiB within 113.04 GiB` (measured with this NVFP4 checkpoint on 2026-10-11; the budget
varies a little from start to start).

Where the memory goes at the default setting (TensorFold's startup estimate; the weights line is the rest of the
98.31 GiB total). This NVFP4 checkpoint's n-gram tables stay memory-mapped on the host (`PLE_ON_SSD=0`; the flag
is for MLX checkpoints and TensorFold refuses it here):

| | GiB |
| --- | ---: |
| Model weights | ~71.0 |
| Vision tower (`VISION=1`) | 0.84 |
| Stream caches (5 x 4.49, the context-sized part) | 22.5 |
| Fixed buffers (DeltaNet states, decode windows, 2,048-row prompt-chunk scratch, 8 saved prompt states) | 4.0 |
| **Startup estimate** | **~98.3** |

The vision tower's scratch (~0.8 GiB at most, measured on a 4,096-token image and a 256-frame video) is taken only
while an image or video encodes and is handed back right after; startup reserves none for it
(`TENSORFOLD_VISION_WORKSPACE_MIB`). With `VISION=0` the tower is not loaded.

Other settings that fit the same budget (rows measured with the earlier MLX 4-bit variant, whose default estimate
was 102.50 GiB; this NVFP4 checkpoint measures ~4 GiB lighter at the default. Trust the server's own admission,
which names a window that fits):

| Setting | KV pool | Estimate | Note |
| --- | ---: | --- | --- |
| `PARALLEL=4` (int8) | 1,048,576 | ~97.7 GiB | more headroom |
| `PARALLEL=5` (int8, default) | 1,310,720 | ~102.5 GiB | |

A setting that does not fit is refused at startup, before any weights load, with a message naming a window that fits.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment
(`PARALLEL=4 ./start.sh`), in a `.env` file next to `start.sh` (`KEY=value` lines, e.g. `PARALLEL=4`; the environment
wins over it), or with `tensorfold serve` flags (`./start.sh --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together (streams) |
| `CONTEXT` | `262144` | prompt + reply window per stream |
| `KV_DTYPE` | `int8` | `bf16`, `int8` or `int4` KV cache |
| `PLE_ON_SSD` | `0` | `1` reads an MLX checkpoint's n-gram tables from SSD instead of RAM; this NVFP4 checkpoint's tables stay memory-mapped (TensorFold refuses `--ple-on-ssd` for it) |
| `VISION` | `1` | image and video input (`--vision`); `0` serves text only |
| `VISION_URLS` | `0` | `1` also accepts public `https://` image and video URLs (default: data URLs only) |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | MTP drafting: at most this many drafts a round, a chain stopping before a draft under the confidence |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling; a request's own values win |
| `THINKING` | `1` | open a think block by default; `0` answers directly unless a request asks to think |
| `MAX_TOKENS` | `32768` | reply length for a request without `max_tokens`; clamped to the stream's window |
| `SERVED_NAME` | `SuperQwen3.8-Flash-Next-abliterated` | the model id in `/v1/models` and in replies |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | where the API listens |
| `VISION_MAX_IMAGES` | `50` | images a request may carry |
| `TENSORFOLD_PREFILL_ROWS` | TensorFold's choice (with `PLE_ON_SSD=1` on an MLX checkpoint: `2048`) | prompt piece rows |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts ahead of MTP (with `PARALLEL >= 2`); `0` for MTP drafts only |
| `TENSORFOLD_IMAGE_TOKENS` | `16384` | tokens a request's images share |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | video token budget |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | `0` | vision tower scratch reserve |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `2` | GiB left out of MemAvailable at admission |
| `PREPARE` | `auto` | `start.sh` runs `scripts/prepare.sh` when needed |
| `PULL` | `1` | `prepare.sh` tries the prebuilt image first; `0` always builds locally |
| `STOP_TIMEOUT` | `30` | seconds `stop.sh` gives the server to shut down |

Any `TENSORFOLD_*` variable in the environment is passed into the container (`TENSORFOLD_NO_UPDATE_CHECK=1`, the
default, stops TensorFold asking GitHub for a newer release at each start).

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** runs `scripts/prepare.sh` whenever the setup is not ready: on the first run, after the patches change,
   or with another model or image. It compares what `prepare.sh` last left ready with the current settings, so later
   starts skip it instantly.
2. **Checks:** the arguments (with TensorFold's own parser, in a throwaway container), the previous server, the
   port and the free memory.
3. **Launch:** `tensorfold serve` with the settings from `scripts/config.sh`.
4. **Loading:** the server's log as it comes, and every 15 s the elapsed time and how much of the startup estimate is
   on the GPU. If the server stops, the last log lines and the reason are shown.
5. **Smoke test:** one chat completion, then the LIVE message and the endpoint.

If the server is already running, `./start.sh` says so and leaves it alone; `./start.sh restart` stops it and
starts it again. It stops the server only after the setup and the argument check pass, so a typo leaves the running
server alone and the server is down only while it restarts. Stopping cuts off requests still running (`stop.sh` warns
when there are any). Extra arguments go to `tensorfold serve` after the defaults, so they win
(`./start.sh restart --context 131072`); `./start.sh --help` lists the options. `FOREGROUND=1 ./start.sh` stays
attached to the server's log and exits with its exit code (for a systemd unit).

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight: Docker, the NVIDIA runtime, disk space.
2. The image `tensorfold-superqwen38-abliterated:v0.6.1`: TensorFold v0.6.1 with every `patches/*.patch` applied,
   plus `transformers` (the vision tower) and PyAV (video decoding), on NVIDIA's PyTorch container
   (`nvcr.io/nvidia/pytorch:26.07-py3`). It first tries the matching prebuilt image from GitHub
   Container Registry (`ghcr.io/piotreknow02/superqwen3.8-abliterated-single-dgx-spark-tensorfold:v0.6.1-<patches hash>`,
   ~11 GB); if that tag is not there (e.g. after you change `patches/`), or with `PULL=0`, it builds the image locally instead.
3. Downloads the checkpoint into `~/.cache/huggingface` (resumable).
4. Verifies the checkpoint with `tensorfold info`.

Run it yourself to download ahead of time or to rebuild the image from scratch:

```bash
scripts/prepare.sh                  # set up without starting the server
scripts/prepare.sh --rebuild         # rebuild the image from scratch
PREPARE=1 ./start.sh restart        # force prepare.sh, then restart
PREPARE=0 ./start.sh restart        # skip prepare.sh on restart
```

After changing `patches/`, `scripts/publish-image.sh` pushes the new image to GitHub Container Registry (`:latest`
and `:v0.6.1-<patches hash>`).

## Credits

This recipe continues [Mia'a AI Lab's original SuperQwen3.8-abliterated DGX Spark recipe](https://github.com/MiaAI-Lab/SuperQwen3.8-abliterated-Single-DGX-Spark-TensorFold);
[`CREDITS.md`](CREDITS.md) names everyone this build stands on. Licensed under the MIT License ([`LICENSE`](LICENSE)).
