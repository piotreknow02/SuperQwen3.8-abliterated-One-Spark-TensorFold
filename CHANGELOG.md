# Changelog

All notable changes to the SuperQwen3.8-Flash-Next-abliterated-DGX-Spark-TensorFold recipe.

## 2026-10-26 (repository handover)

- The repository is now maintained by piotreknow02; the recipe was originally developed by Mia'a AI Lab (see `CREDITS.md`).
  Prebuilt images move to `ghcr.io/piotreknow02/superqwen3.8-abliterated-single-dgx-spark-tensorfold`
  (`GHCR_IMAGE` in `scripts/config.sh`); while no matching tag is public, `scripts/prepare.sh` builds the image locally.
- Removed the `start.sh` terminal banner, the pre-v0.6.1 development logs (`docs/astra*.md`) and the
  `TF_SRC` patch tests (`tools/test_*.py`), which described an earlier patch series; they documented the previous
  author's optimization work, not what this recipe ships.
- Serving settings, `patches/0002-flash-next-v061.patch` and the measured numbers are unchanged.

## 2026-10-11 (NVFP4 checkpoint)

- Switched `MODEL_ID` to `Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark`: TensorFold v0.6.1's CUDA
  kernels read NVFP4 (ModelOpt) or MLX 4-bit group-32 only, so the original MLX 4-bit group-64 checkpoint could not
  load on a DGX Spark (`CUDA_QUANTIZATION = (4, 32)`, hard error at weight load). The NVFP4 variant is the same
  abliterated model, converted with the sealed TensorRT-Model-Optimizer revision (~136 GiB, MTP head retained).
- MTP drafting on, matching the base-model recipe: `MTP_DRAFTS=6`, `MTP_CONFIDENCE=0.60`, `TENSORFOLD_MTP_COPY=1`
  (`--no-drafts` removed).
- `MIN_FREE_GB` 125 -> 145 (the checkpoint is ~136 GB).
- `PLE_ON_SSD` default `1` -> `0`: TensorFold refuses `--ple-on-ssd` for NVFP4 checkpoints (their n-gram tables
  stay memory-mapped on the host); the flag remains for MLX checkpoints.
- Measured admission (2026-10-11): the default 5 x 262,144 int8 fits, startup estimate 98.31 GiB within 113.04 GiB.
- Same `patches/0002-flash-next-v061.patch` (same Qwen4Exp architecture); no image change.

## 2026-10-08 (initial)

- Adapted from `Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` recipe (TensorFold v0.6.1).
- Models `Jiunsong/SuperQwen3.8-Flash-Next-abliterated-MLX-4bit` (MLX 4-bit, group size 64, no MTP draft head).
- `--no-drafts` is always used; `MTP_DRAFTS` / `MTP_CONFIDENCE` are not applicable.
- `DRAFT_LANGUAGE` support removed (no MTP draft vocabulary for this model).
- `TENSORFOLD_MTP_COPY` forced to `0` (no MTP to copy from).
- Image: `tensorfold-superqwen38-abliterated`; GHCR: `ghcr.io/miaai-lab/superqwen3.8-abliterated-single-dgx-spark-tensorfold`.
- Container: `superqwen38-abliterated-tf`.
- API model name: `SuperQwen3.8-Flash-Next-abliterated`.
- Checkpoint download: ~114 GiB (MLX 4-bit quantized).
- Same `patches/0002-flash-next-v061.patch` applies (same Qwen4Exp architecture).
- Same performance profile as the Vontra MTP variant (same architecture, no MTP overhead).