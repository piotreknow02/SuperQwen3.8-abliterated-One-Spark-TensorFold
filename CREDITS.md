# Credits

This repository is maintained by **piotreknow02**. It continues and adapts **Mia'a AI Lab**'s original DGX Spark recipe.

- **Mia'a AI Lab** ([GitHub](https://github.com/MiaAI-Lab), [X](https://x.com/MiaAI_lab)) for the original DGX Spark recipe and the TensorFold patches this one started from ([SuperQwen3.8-abliterated-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/SuperQwen3.8-abliterated-Single-DGX-Spark-TensorFold)).
- **Jiunsong** for the [`SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark`](https://huggingface.co/Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark) NVFP4 quantized checkpoint (the same abliterated model as the MLX 4-bit packaging, in the format TensorFold's CUDA kernels read).
- **Qwen** for the upstream [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) architecture the abliterated model is based on.
- **ashhart** for [TensorFold](https://github.com/ashhart/TensorFold), the serving framework.
- **NVIDIA** for [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) used to convert the sealed model to NVFP4.
