# Historical Benchmark Protocol and Reproduction Scope

## Workload

Four fixed SFT checkpoints: standard full-parameter, GenLoRA, OLS + steepest descent, and full-parameter Hopfield with the authors' equations ported into this architecture. Every path receives the same 288 questions in the same arrival order: 256 domain questions and 32 everyday questions, over three rounds. Domain questions paraphrase 64 learned facts; everyday questions cover 16 tasks. This speed workload is not an unseen-knowledge accuracy benchmark.

## Timing

- Exclude common model loading and warm-up. Include method-specific compilation, trial execution, Capture, timing probes and any waiting that blocks requests.
- Begin each round with empty online history. Do not pre-capture graphs or choose buckets from the question list; update history only after requests complete.
- Alternate or rotate method order according to each baseline/online experiment's protocol. Do not present separate batches as one randomized experiment.
- Time to first token: from request entry to delivery of the first answer token. Total latency: until stopping. Decode tokens/s covers generation after the first token only.
- Divide each round's total elapsed time by 288, then average over three rounds for the round-level mean. This includes preparation costs and cannot be replaced by steady-state P50.
- For each question and metric, first take the median across three rounds, then report P50/P95 separately for the 256 domain and 32 everyday questions. These are not extra runs.
- Report the cold first request separately without subtracting it from the round total. Check Hopfield cold JIT in a fresh process as well.
- Check complete token sequences, stop reasons and EOS for every output. Retain immediate stops, length-limit stops and failures; do not shorten answers to claim speed improvements.

## Device and Precision

Historical formal experiments used one NVIDIA L20. Some execution paths differ in FP16/FP32 settings; read the path descriptions in the [results](RESULTS.md). Token agreement was checked on the frozen workload and does not guarantee identical mixed-precision output for arbitrary new inputs.

## What This Release Reproduces

CPU tests cover model shapes, causality, optimizers, three SFT routes, updates/reloading/resume, tokenization and retrieval. Example commands train a small model end to end. Formal weights, complete source corpora and private raw run records are not released; a two-step demo cannot reproduce the historical latency. After training your own model, save environment/device details and checkpoint/input/source hashes, then compare reference and online paths using these rules. Do not label your new measurements as the historical experiment.
