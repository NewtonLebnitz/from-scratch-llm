# 运行手册

手写中英字节 BPE、GQA Transformer、保留跨层注意力状态的 Hopfield 变体、AdamW 预训练与三种生成式监督微调。包含生成、PCA/SOM 检索核心和按已到达请求选择 CUDA Graph 的在线分桶源码。

## 运行环境与安装

- 基础入口在 Windows、Linux 使用 Python 3.12 或 3.13。CPU 可以运行小模型、验证训练和生成流程。
- 使用 `requirements.txt` 中的 PyTorch、NumPy、regex。CPU 验证的具体版本见发布验证记录。全规模训练请按可用资源调整配置。
- CUDA 在线分桶的原实验环境为 Linux、Python 3.13、PyTorch 2.12.1（CUDA 13.0 构建）。需 NVIDIA GPU、匹配的驱动、支持 CUDA Graph 的 PyTorch。Hopfield 融合内核还需对应 PyTorch 发行版所配套的 Triton，且注意力头宽度须为 64。
- 需要多语 E5 文本编码器时，按 `requirements-retrieval.txt` 安装可选依赖。纯文本生成、普通监督微调和直接使用向量训练 SOM 不依赖它们。

在仓库根目录执行。

```text
python -m venv .venv
```

PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`，Linux 为 `source .venv/bin/activate`。激活后执行：

```text
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests
python -X utf8 -B main.py --help
```

GPU 运行者应安装适配驱动的 PyTorch CUDA 发行版，设置 `CUDA_VISIBLE_DEVICES` 使程序仅能看到一块已获准的 GPU，再在命令后添加 `--device cuda`。支持 BF16 的设备训练时可额外指定 `--bf16`。CLI 默认使用 CPU。

## 输入材料与外部路径

以 `.runtime/run` 为运行工作区。模型、分词器、SOM 索引和生成数据都保存在此工作区；原始文本和问答也由使用者另外准备。本仓库提供程序、配置和说明。可先执行 `python -B scripts/make_demo_data.py --output .runtime/input` 创建原创合成小样例；这些样例仅验证链路，不能用于衡量问答泛化。

- **预训练文本**可从 [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) 的中英文分片或其他已获授权的语料准备。去重后按文档或来源族隔离 TRAIN 与 DEV。在 `.runtime/input/pretrain/` 放置 `zh-train.jsonl`、`zh-dev.jsonl`、`en-train.jsonl`、`en-dev.jsonl`。每行是一个 JSON 对象，`text` 字段存该文档正文。每个文件的有效 token 数须大于配置里的序列长度。
- **SFT 问答**由使用者审核后整理为 UTF-8 JSONL。每行须有字符串 `question`、`answer`、`family_id`，以及非空的来源 ID 列表 `source_ids`。可选 `context`、`system`。同一来源族不能横跨 TRAIN、DEV、TEST。放入 `.runtime/input/sft/train.jsonl` 和 `dev.jsonl`。如有独立 TEST，以 `--test` 注册。历史工程中的课程语料、论文与答案均需由使用者另行准备。
- **预训练检查点**由下面的预训练命令生成。若已有可信的本项目检查点，放在工作区的 `artifacts/models/<名称>/`，匹配分词器放在 `artifacts/tokenizers/<名称>/tokenizer.json`。SFT 检查点会校验其基座的相对路径和 SHA-256，复用权重时应保留这些相对路径和相同的分词器。通用 Hugging Face 语言模型权重与本项目检查点格式不同。

## 从头训练与生成

下面的 CPU 小配置主要用来验证链路。它需要中英文本量足够学到 512 项词表，并满足上述最短长度。命令在仓库根目录执行。

```text
python -X utf8 -B main.py --workspace .runtime/run prepare --corpus .runtime/input/pretrain --name demo --vocab-size 512
python -X utf8 -B main.py --workspace .runtime/run pretrain --config configs/pretrain-demo.json --name demo-base
python -X utf8 -B main.py --workspace .runtime/run generate --kind base --checkpoint artifacts/models/demo-base/best.pt --tokenizer demo --prompt "The robot" --max-new-tokens 8
python -X utf8 -B main.py --workspace .runtime/run register-sft --name demo-qa --tokenizer demo --train .runtime/input/sft/train.jsonl --dev .runtime/input/sft/dev.jsonl
python -X utf8 -B main.py --workspace .runtime/run sft --config configs/sft-full-demo.json --name demo-full
python -X utf8 -B main.py --workspace .runtime/run generate --checkpoint artifacts/models/demo-full/best.pt --tokenizer demo --prompt "What is a robot?" --max-new-tokens 8
```

`prepare` 只用 TRAIN 文档学习 BPE，并分别生成四份 uint16 token 流。训练会创建 `best.pt` 和 `last.pt`，预训练输出 DEV CE，SFT 在回答 token 上计算损失。生成输出 JSON，包括真实生成的文本、token ID、停止原因和时间。CPU 小配置只训练少量步数，适合检查流程和张量形状，问答质量取决于数据和训练预算。

- `configs/sft-genlora-demo.json` 和 `configs/sft-ols-demo.json` 可替换全参 SFT 配置。对应输出名改为 `demo-genlora` 或 `demo-ols`。前者更新 RBF 生成的低秩适配器，后者用 TRAIN 答案误差选择 Gaussian 中心，再用最速下降。
- `configs/pretrain-hopfield-demo.json` 可替换标准预训练配置。若输出使用新名称，相应 SFT 配置中的 `base_checkpoint` 要改成这个新检查点。
- `configs/pretrain-standard-18l.json` 和 `configs/pretrain-hopfield-18l.json` 提供 18 层、768 维、32,000 词表的结构及训练预算。使用它们前，将 `prepare` 的词表大小改为 32000、名称改为 `bilingual`，并根据设备容量调整批大小和累积步数。

## 在线 CUDA Graph 推理与 SOM

- 已训练的 SFT 模型可在 `generate` 命令加 `--device cuda --backend cuda-online`。用 `--requests <外部 JSONL>` 代替单个 `--prompt`，每行提供 `question` 与可选 `context`。同一进程按到达顺序维护图缓存，才能观察预填充与生成图的复用。带角色标记的提示加回答预算最多 512 token。FP16 与 FP32 的输出可能因精度或输入而不同，检查指定模型时请用相同输入和参数与参考入口逐条比较。
- `src/mlsystem/retrieval.py` 提供 `fit_som`、`build_index` 和 `SOMIndex`。可传入自行准备的浮点向量矩阵训练 PCA + SOM。建立文档索引时另外提供与向量同行的 `id`、`source_id`、`text`。若使用内置 `MultilingualEncoder`，从 [多语 E5 官方发布页](https://huggingface.co/intfloat/multilingual-e5-small) 准备模型，在工作区 `artifacts/encoders/multilingual-e5-small/` 放入 tokenizer、模型配置和 safetensors；按源码约定提供 `status=ready`、`model=intfloat/multilingual-e5-small` 与含相对路径及 SHA-256 的 `files` 清单。检索出的文本可作为 `generate --context` 的证据，答案仍由模型生成。

## 源码入口

| 功能 | 位置 |
| --- | --- |
| 预训练、监督微调、推理命令 | `main.py` |
| BPE、模型、优化器 | `src/mlsystem/bpe.py`、`model.py`、`optim.py` |
| 三种 SFT、Gaussian OLS、SOM | `src/mlsystem/sft.py`、`genlora.py`、`ols_rbf.py`、`retrieval.py` |
| 在线图准入与原生捕获 | `inference_cuda/online_cuda_policy_v2.py`、`online_cuda_engine_v2.py` |

结构设计参考 Stanford CS336 课程和现代 Hopfield 注意力工作。分词、语言模型、微调和推理由本仓库源码实现。第三方运行依赖按清单安装。


## 核心验证范围

使用独立生成的中英文小样例，在 CPU 上实跑分词、标准与 Hopfield 预训练、标准基座的三种 SFT、Hopfield 全参 SFT，以及各检查点加载后的逐 token 生成。28 项单元测试通过。小样例用于检查流程，CUDA 在线推理入口仅做导入检查，完整训练和 CUDA 推理需在满足环境要求的设备上运行。
