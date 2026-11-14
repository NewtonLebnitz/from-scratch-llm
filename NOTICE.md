# 实现与参考

作者：谢邹健（Xie Zoujian）。本仓库来自课程工程的便携核心，模型训练、微调和推理入口以仓库源码为准。训练规模、性能结果及公开版验证范围见README与docs。

- Stanford CS336：[lectures](https://github.com/stanford-cs336/lectures)、[assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。用于课程设计和基础组件学习。
- 现代Hopfield注意力：Masumura与Taki，NeurIPS2025，[论文](https://arxiv.org/abs/2511.20698)，[作者补充代码](https://proceedings.neurips.cc/paper_files/paper/2025/file/634ddc254c9e739eea6c5393e3abe5c9-Supplemental-Conference.zip)。当前实现移植其分数状态与混合机制，另配本项目的RMSNorm/RoPE/GQA/SwiGLU架构。
- RBF生成低秩适配器相关工作：[参考论文](https://arxiv.org/abs/2605.30385)。仓库为当前模型上的手写实现，不声称作者官方实现。
- [PyTorch](https://pytorch.org/)与[Triton](https://triton-lang.org/)为外部运行依赖，各自许可证适用于相应软件；可选文本检索使用[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)。

本仓库不分发上述第三方预训练权重或论文全文，不包含RA-L在投论文研究代码或研究内容。
