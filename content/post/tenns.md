+++
title = "Temporal Neural Networks (TENNs) at BrainChip"
date = "2025-01-15"
tags = ["state-space-models", "edge-ai", "deep-learning"]
+++

While working at [BrainChip](https://brainchip.com/), I led the research and software-hardware codesign of TENNs (Temporal Neural Networks) — a family of deep state-space models designed for efficient, low-latency inference on edge hardware. This line of work developed in parallel with the broader SSM wave (Mamba, Gated Delta Net, etc.), cross-pollinating ideas around structured recurrences, parallel scans, and kernel parameterizations.

<!--more-->

The project resulted in several publications:

- [**TENNsEye**](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Pei_A_Lightweight_Spatiotemporal_Network_for_Online_Eye_Tracking_with_Event_CVPRW_2024_paper.html) (CVPR 2024 Workshop) — A lightweight causal spatiotemporal network for online eye tracking with event cameras. Targets edge hardware via simple operations (convolutions, ReLU), online inference through output buffering, and >90% activation sparsity through regularization. Reached 0.9916 p10 accuracy on the AIS 2024 challenge.

- [**Centaurus**](https://www.youtube.com/watch?v=Ajo3RIk5y5M&t=59s) (ICLR 2025 Spotlight, solo author) — Treats SSM operations as tensor contractions and systematically determines their optimal ordering to maximize training efficiency. This unlocks flexible SSM block designs beyond depthwise-separable configurations — group convolutions, full convolutions, bottleneck blocks — all composed into a heterogeneous architecture. The first fully state-space network with competitive ASR performance without LSTMs, explicit convolutions, or attention.

- [**aTENNuate**](https://www.isca-archive.org/interspeech_2025/pei25_interspeech.pdf) (Interspeech 2025) — A deep state-space autoencoder for real-time raw speech enhancement. Processes waveforms end-to-end with no spectral pre/post-processing, outperforming prior real-time denoising models in PESQ while using only 0.84M parameters and 0.33G MACs. Remains performant even when input is compressed down to 4000 Hz and 4 bits.

- [**PLEIADES**](https://www.youtube.com/watch?v=B5bzYl4zjPU&t=401s) (NeurIPS 2025) — Uses structured temporal kernels based on orthogonal polynomials for online spatiotemporal classification and detection. Handles variable sample rates and discretization step-sizes without fine-tuning. Achieved state-of-the-art on DVS128 gesture recognition (99.6%), AIS 2024 eye tracking (99.6%), and PROPHESEE 1MP automotive detection (0.556 mAP) — all with sub-million parameter counts.

On the side, I also put together [**mamba-tiny**](https://github.com/PeaBrane/mamba-tiny) — a minimal single-file PyTorch implementation of the Mamba SSM using `logcumsumexp` as an alternative to the parallel scan (which wasn't available in PyTorch at the time).

[Back to projects](/post)
