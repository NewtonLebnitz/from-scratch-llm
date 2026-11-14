# Release validation

Checked on 2026-11-14. These checks validate the released source and small-model execution, not the quality of a model trained for two steps.

| Check | Result |
|---|---|
| Local unit tests | 28 passed |
| README quick start | Fixture generation → BPE → pretraining → full SFT → checkpoint reload and generation passed |
| Other standard SFT routes | GenLoRA and OLS each trained, reloaded and generated tokens |
| Hopfield route | Hopfield pretraining and full SFT each completed; SFT checkpoint reloaded and generated tokens |
| Public CI | GitHub Actions CPU test job passed on the initial release |
| Source audit | Python syntax, JSON, relative Markdown links, private paths and credential signatures checked |

Local runtime: Windows, Python 3.12.3, PyTorch 2.13.0+cpu; `CUDA_VISIBLE_DEVICES` empty; two CPU math threads. All generated data, checkpoints and caches are excluded from version control. No GPU experiment was rerun for this release. Historical GPU performance measurements and their separate protocols are in [RESULTS.md](RESULTS.md) and [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

The small demo corpus is original synthetic text. Its train/dev examples intentionally serve execution checks, not independent knowledge generalization. Use source/semantic-family separation and a separately frozen evaluation set for real quality experiments.
