# Shared settings for start.sh, stop.sh and scripts/*.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh` or `PULL=0 scripts/prepare.sh`, or set in ./.env: KEY=value lines, read here (never
# run as a script); a variable already set in the environment wins over the file. .env is yours, not the repository's.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi

MODEL_ID="${MODEL_ID:-Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark}"   # NVFP4 (ModelOpt), blocks of 16, with the MTP head
# The patches and start.sh's flags are made for TensorFold v0.6.1 exactly (17c73e1). After changing
# TF_VERSION, TF_REPO or BASE_IMAGE, run `scripts/prepare.sh --rebuild`.
TF_VERSION="${TF_VERSION:-v0.6.1}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-superqwen38-abliterated}"   # the local image prepare.sh builds or pulls
CONTAINER_NAME="${CONTAINER_NAME:-superqwen38-abliterated-tf}"    # the server's container
# The prebuilt images: prepare.sh pulls $GHCR_IMAGE:<TF_VERSION>-<patches hash>; publish-image.sh pushes it (and
# :latest), published from this repository by scripts/publish-image.sh; until a matching tag is public,
# prepare.sh builds the image locally. This recipe bakes patches/*.patch only.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/piotreknow02/superqwen3.8-abliterated-single-dgx-spark-tensorfold}"

SERVED_NAME="${SERVED_NAME:-SuperQwen3.8-Flash-Next-abliterated}"   # the model id clients see in /v1/models and replies
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# Serving defaults (./start.sh arguments come after them and win). All streams share one memory pool, so window x streams
# x KV bytes must fit. Measured with this NVFP4 checkpoint (2026-10-11): 5 x 262,144 int8 is admitted, startup
# estimate 98.31 GiB within 113.04 GiB. TensorFold refuses a window that does not fit, naming one that does.
PARALLEL="${PARALLEL:-5}"          # requests decoded together (streams)
CONTEXT="${CONTEXT:-262144}"       # prompt + reply window per stream (the model's native maximum)
KV_DTYPE="${KV_DTYPE:-int8}"       # bf16 | int8 | int4
# MTP drafting: at most MTP_DRAFTS drafts a round, a chain stopping before a draft under MTP_CONFIDENCE.
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
# 0: this NVFP4 checkpoint's n-gram tables stay memory-mapped on the host (TensorFold refuses --ple-on-ssd for
# NVFP4); 1: an MLX checkpoint's tables are read from SSD instead, leaving that RAM to the KV cache.
PLE_ON_SSD="${PLE_ON_SSD:-0}"
# Image input (TensorFold's Flash Next vision, patch 0002): the model's own vision tower from the same checkpoint.
# The vocabulary includes the vision tokens even if VISION=0. VISION=0: text only, skips loading the vision tower.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"    # 1: also accept public https:// image and video URLs (default: data URLs only)
# Images a request may carry (--vision-max-images; a chat's turns all count). They share TENSORFOLD_IMAGE_TOKENS.
# Empty: TensorFold's own limit (4).
VISION_MAX_IMAGES="${VISION_MAX_IMAGES-${TENSORFOLD_MAX_IMAGES:-50}}"
# Thinking mode (Qwen's recommended sampling): temperature 1.0, top_p 0.95, top_k 20. A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
# Reply length for a request that sets no max_tokens. TensorFold's own default, 4,096, can end a thinking reply before
# it answers. The value is clamped to the room left in the stream's window.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# TensorFold switches from the environment (TENSORFOLD_*) reach the server too.
# Prompt-lookup drafts ahead of MTP (patch 0002's copy drafts; with PARALLEL >= 2): +6% on replies that repeat the
# prompt, prose and tool calls (measured with the MTP checkpoint). 0: MTP drafts only.
export TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
# Prompt piece rows. Unset, TensorFold picks 2,048 with vision, 4,096 without (while nothing decodes, if memory
# allows). With the n-gram tables on SSD (PLE_ON_SSD=1) 4,096 measured 10-30% slower from 5k to 16k tokens and
# ~3% slower at 31k (2026-10-02), so this recipe keeps 2,048 there. TENSORFOLD_PREFILL_ROWS=N (256 to 16,384,
# patch 0002) forces N-row pieces, admitted with the window; empty: TensorFold's choice.
if [[ "$PLE_ON_SSD" == 1 ]]; then export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS-2048}"; fi
# What startup reserves for the vision tower's scratch (MiB); 0: it comes from the system reserve while it encodes.
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"
# The tokens a request's images share (VISION_MAX_IMAGES above), each image at most 4,096. The tower encodes them
# 16,384 patches at a time, the scratch one image needs.
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
# The whole video's token budget (Qwen3-VL's per-frame sizing; 2 frames a second, at most 256 frames).
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
# Startup reserve (since v0.6.0): GiB left out of MemAvailable. Unset, TensorFold takes max(4 GiB, a tenth of RAM)
# and refuses 5 x 262,144. The knob's floor is 2 GiB; that still fits the measured ~102.5 GiB admission.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
# No "is there a newer TensorFold" call to GitHub at each start: the patches are for v0.6.1 anyway. 0: check.
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Persists compiled CUDA kernels (torch extensions + triton) so only the first start pays the compile.
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-superqwen38-abliterated}"

MIN_FREE_GB="${MIN_FREE_GB:-145}"   # free disk the checkpoint download needs (it is ~136 GB)
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"   # free disk under Docker's root that pulling or building the image needs

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }

# The patches baked into $IMAGE, in order: patches/*.patch only.
patch_files() { ls patches/*.patch 2>/dev/null; }
patches_hash() { patch_files 2>/dev/null | xargs -r cat | sha256sum | cut -c1-12; }

# What scripts/prepare.sh last left ready (it writes this line to PREPARED_MARKER when it succeeds); start.sh runs
# prepare.sh again whenever the current line differs: a missing or stale image, new patches, another model.
PREPARED_MARKER="$KERNEL_CACHE/.prepared"
prepared_state() {
  local hash label model=missing
  hash=$(patches_hash)
  label=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || echo missing)
  ls -d "$(model_cache_dir)"/snapshots/*/ >/dev/null 2>&1 && model=present
  echo "model=$MODEL_ID($model) image=$IMAGE($label) patches=$hash"
}
