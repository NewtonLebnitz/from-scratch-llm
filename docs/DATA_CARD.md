# Data Card

## Included in the Repository

`scripts/make_demo_data.py` generates original bilingual synthetic documents and a small number of QA examples for CPU preparation, pretraining, SFT and generation checks. These are execution fixtures, not a knowledge-capability benchmark. Unicode escapes preserve Chinese fixture text while keeping the source readable in English.

## Historical Full-Size Task

- Bilingual pretraining uses processed FineWiki documents. The 32k BPE vocabulary is fitted on TRAIN text only.
- SFT uses 14,336 reviewed bilingual QA records covering limited facts from VLA materials and papers available in full text, as well as everyday tasks.
- Inference evaluation separates 256 domain questions and 32 everyday questions. Domain paraphrases concern learned facts; this does not imply coverage of all VLA knowledge or all papers by an author.

## Materials Not Released

Full papers, source texts subject to their own terms, complete training corpora, trained checkpoints, internal service configurations and unpublished VLA research are not included. Obtain authorized data and use the formats in `docs/USAGE.md`.

## Splitting Principles

For new tasks, separate TRAIN and DEV/TEST by document, source family or semantic group to prevent paraphrase leakage. DEV selects checkpoints; TEST is not used for tuning. Data files, tokenizers and parent models are bound by SHA-256. Store model answers separately from retrieved evidence.
