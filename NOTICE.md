# Implementation and References

Author: Xie Zoujian. This repository contains the portable core of a course engineering project. The source defines the model training, fine-tuning and inference entry points. See the README and documentation for training scale, measured performance and release validation.

- Stanford CS336: [lectures](https://github.com/stanford-cs336/lectures) and [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics), used for course design and learning fundamental components.
- Modern Hopfield attention: Masumura and Taki, NeurIPS 2025, [paper](https://arxiv.org/abs/2511.20698) and [authors' supplementary code](https://proceedings.neurips.cc/paper_files/paper/2025/file/634ddc254c9e739eea6c5393e3abe5c9-Supplemental-Conference.zip). This implementation ports the score-state and mixing mechanism into this project's RMSNorm/RoPE/GQA/SwiGLU architecture.
- RBF-generated low-rank adapters: [reference paper](https://arxiv.org/abs/2605.30385). The repository implements the method for its own model; it is not the authors' official implementation.
- [PyTorch](https://pytorch.org/) and [Triton](https://triton-lang.org/) are external runtime dependencies governed by their respective licenses. Optional text retrieval uses [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

This repository does not redistribute third-party pretrained weights or full papers, and does not contain the code or research content of the RA-L submission.
