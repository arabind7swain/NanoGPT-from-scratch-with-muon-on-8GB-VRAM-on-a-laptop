# NanoGPT-from-scratch-with-muon

This repository contains a highly optimized training recipe to "speedrun" a GPT-2 124M-class model to < 3.28 validation loss (the OpenAI baseline) using a single consumer laptop GPU.

The target: 3.28 Validation Loss in under 180 minutes on an RTX 4050 mobile arvhitecture. Currently it takes about 4.5 hrs.
🛠 Features & Architecture

We use a "Modded-NanoGPT" architecture that deviates from the 2019 original to maximize hardware utilization:

    Muon Optimizer: A Newton-Schulz-style optimizer that orthogonalizes matrix parameters, leading to significantly faster convergence than AdamW.

    U-Net Skip Connections: Enhances gradient flow by connecting early encoder layers to later decoder layers.

    Value Embeddings: Triple-stream value_embeds tables that add model capacity without increasing FLOPs.

    FlexAttention: Leverages torch.nn.attention.flex_attention with document-causal masking for superior memory efficiency.

    Modern Stabilizers: Rotary Positional Embeddings (RoPE) and Tanh Logit Scaling (30×tanh(x/30)) to prevent FP8/BF16 overflows.

💻 Hardware Optimization (RTX 5070 Laptop)

To run this on a laptop with 8GB VRAM (standard for 5070 Mobile):

    FP8 Training: Enabled via torchao. This is critical for the 50-series Tensor cores.

    Progressive Windowing: Training starts at a 256 context length and scales to 1792, saving massive compute in the early phase.

    Thermal Management: The script includes async data loading to keep the GPU saturated while minimizing CPU-side overhead that causes laptop throttling.
