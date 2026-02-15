+++
title = "NVIDIA Dynamo: Distributed LLM Inference"
date = "2025-10-01"
tags = ["llm-inference", "distributed-systems", "rust"]
+++

[Dynamo](https://github.com/ai-dynamo/dynamo) is NVIDIA's open-source datacenter-scale distributed inference serving framework for generative AI and reasoning models. Built in Rust for performance and Python for extensibility, it supports disaggregated prefill and decode, dynamic GPU scheduling, and LLM-aware request routing across multi-node multi-GPU topologies. The project has 6k+ GitHub stars and supports backends including TensorRT-LLM, vLLM, and SGLang.

<!--more-->

I led the efforts in developing three core components:

- **KV-aware router** — Routes and load-balances requests to workers while eliminating unnecessary KV cache recomputation, leveraging prefix caching coordination via NATS.

- **Engine mockers** — Simulated inference engine backends for testing and benchmarking the serving infrastructure without requiring actual GPU resources.

- **Data synthesizer** — Synthetic workload generation for stress-testing and evaluating the serving pipeline under realistic traffic patterns.

[Back to projects](/post)
