# Usage Guide

From-scratch bilingual byte-level BPE, GQA Transformer, a Hopfield variant with cross-layer attention state, AdamW pretraining and three generative SFT routes. The release includes generation, PCA/SOM retrieval and arrival-history-based online CUDA Graph bucketing.

## Environment and Installation

- Use Python 3.12 or 3.13 on Windows or Linux. CPU supports the small-model training and generation checks.
- Install PyTorch, NumPy and regex from `requirements.txt`. See [release validation](VALIDATION.md) for the checked CPU versions. Adjust full-size configurations to your hardware.
- Historical online CUDA experiments used Linux, Python 3.13 and PyTorch 2.12.1 built with CUDA 13.0. You need an NVIDIA GPU, a compatible driver and CUDA Graph support. Hopfield fused kernels require Triton matched to the PyTorch distribution and attention head width 64.
- Install `requirements-retrieval.txt` for the optional multilingual E5 encoder. Generation, standard SFT and SOM training directly from vectors do not require it.

Run from the repository root:

```text
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux, then run:

```text
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests
python -X utf8 -B main.py --help
```

For GPU use, install the appropriate PyTorch CUDA build, set `CUDA_VISIBLE_DEVICES` to expose a single GPU you are authorized to use, and add `--device cuda`. Training on BF16-capable hardware can additionally use `--bf16`. CPU is the CLI default.

## Inputs and Workspace

Use `.runtime/run` for generated data, tokenizers, models and SOM indexes. Supply source documents and QA separately. Run `python -B scripts/make_demo_data.py --output .runtime/input` for original synthetic fixtures. These validate execution, not QA generalization.

- **Pretraining text:** prepare English/Chinese [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) shards or another authorized corpus. Deduplicate and separate TRAIN/DEV by document or source family. Place `zh-train.jsonl`, `zh-dev.jsonl`, `en-train.jsonl`, `en-dev.jsonl` in `.runtime/input/pretrain/`. Each line is a JSON object with document content in `text`. Each file needs more valid tokens than the configured sequence length.
- **SFT QA:** reviewed UTF-8 JSONL records with string fields `question`, `answer`, `family_id` and a nonempty list `source_ids`; optional `context` and `system`. A source family must not cross TRAIN/DEV/TEST. Use `.runtime/input/sft/train.jsonl` and `dev.jsonl`; register an independent TEST with `--test` if available. Prepare the historical course materials, papers and answers separately.
- **Pretrained checkpoints:** produced by the commands below. Existing trusted project checkpoints go under workspace `artifacts/models/<name>/`, with their tokenizer at `artifacts/tokenizers/<name>/tokenizer.json`. SFT checkpoints verify the relative path and SHA-256 of the parent base. Preserve these paths and the same tokenizer when reusing weights. Generic Hugging Face LM weights use a different format.

## Training and Generation

The small CPU configuration verifies the pipeline. Provide enough bilingual text to learn a 512-entry vocabulary and satisfy the minimum token lengths. Run from the repository root:

```text
python -X utf8 -B main.py --workspace .runtime/run prepare --corpus .runtime/input/pretrain --name demo --vocab-size 512
python -X utf8 -B main.py --workspace .runtime/run pretrain --config configs/pretrain-demo.json --name demo-base
python -X utf8 -B main.py --workspace .runtime/run generate --kind base --checkpoint artifacts/models/demo-base/best.pt --tokenizer demo --prompt "The robot" --max-new-tokens 8
python -X utf8 -B main.py --workspace .runtime/run register-sft --name demo-qa --tokenizer demo --train .runtime/input/sft/train.jsonl --dev .runtime/input/sft/dev.jsonl
python -X utf8 -B main.py --workspace .runtime/run sft --config configs/sft-full-demo.json --name demo-full
python -X utf8 -B main.py --workspace .runtime/run generate --checkpoint artifacts/models/demo-full/best.pt --tokenizer demo --prompt "What is a robot?" --max-new-tokens 8
```

`prepare` fits BPE on TRAIN only and creates four uint16 token streams. Training saves `best.pt` and `last.pt`; pretraining reports DEV cross-entropy, while SFT computes loss only on answer tokens. Generation returns JSON with generated text, token IDs, stopping reason and timing. Tiny configurations verify execution and tensor shapes; answer quality depends on data and budget.

- Replace the full SFT config with `configs/sft-genlora-demo.json` or `configs/sft-ols-demo.json`, and use output names `demo-genlora` or `demo-ols`. GenLoRA updates RBF-generated low-rank adapters; OLS selects Gaussian centers using TRAIN answer errors and then applies steepest descent.
- Replace the standard pretraining config with `configs/pretrain-hopfield-demo.json` for Hopfield. If you change the output name, update the SFT config's `base_checkpoint` accordingly.
- `configs/pretrain-standard-18l.json` and `configs/pretrain-hopfield-18l.json` define the 18-layer, 768-dimensional, 32,000-vocabulary architecture and pretraining budget. Before using them, run `prepare` with vocabulary 32000 and name `bilingual`; adjust batch size and accumulation to your device.

## Online CUDA Graph Inference and SOM

- For a trained SFT checkpoint, add `--device cuda --backend cuda-online` to `generate`. Use `--requests <external JSONL>` instead of one `--prompt`; each line provides `question` and optional `context`. The process maintains graph history in arrival order, allowing prefill/decode reuse. Role-marked prompt plus answer budget is limited to 512 tokens. FP16 and FP32 can differ; compare the same inputs and parameters against the reference path per output.
- `src/mlsystem/retrieval.py` provides `fit_som`, `build_index` and `SOMIndex`. Supply a floating-point vector matrix for PCA/SOM, and aligned document `id`, `source_id`, `text` for indexing. For `MultilingualEncoder`, obtain [multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small); place tokenizer, model configuration and safetensors under workspace `artifacts/encoders/multilingual-e5-small/`. Supply the manifest required by the source: `status=ready`, `model=intfloat/multilingual-e5-small` and `files` with relative paths and SHA-256. Retrieved text can be passed as `generate --context`; the model still generates the answer.

## Source Entry Points

| Function | Location |
|---|---|
| Pretraining, SFT and generation CLI | `main.py` |
| BPE, model and optimizer | `src/mlsystem/bpe.py`, `model.py`, `optim.py` |
| Three SFT routes, Gaussian OLS and SOM | `src/mlsystem/sft.py`, `genlora.py`, `ols_rbf.py`, `retrieval.py` |
| Online graph admission and native Capture | `inference_cuda/online_cuda_policy_v2.py`, `online_cuda_engine_v2.py` |

Architecture design references Stanford CS336 and modern Hopfield attention. Repository sources implement tokenization, the language model, fine-tuning and inference; install external dependencies from the requirements files.

## Validation Scope

Original small bilingual fixtures were used for CPU tokenization, standard/Hopfield pretraining, three standard SFT routes, Hopfield full SFT and generation after reloading every checkpoint. All 28 unit tests passed. These examples verify execution. The CUDA entry was import-checked; full-size training and CUDA inference require suitable hardware and dependencies.
