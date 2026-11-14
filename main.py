"""Portable entry points for the handwritten model. Runtime files live in --workspace."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from mlsystem.data import SPECIAL_TOKENS, clean_document, iter_jsonl


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def name(text):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', text):
        raise argparse.ArgumentTypeError('Use letters, digits, dashes or underscores for the artifact name.')
    return text


def prepare(root, args):
    import numpy as np
    from mlsystem.bpe import Tokenizer, train_bpe_documents

    corpus = args.corpus.resolve()
    splits = [(lang, split) for lang in ('zh', 'en') for split in ('train', 'dev')]
    paths = {(lang, split): corpus / f'{lang}-{split}.jsonl' for lang, split in splits}

    def documents(path):
        for row in iter_jsonl(path):
            text = clean_document(row['text'])
            if text:
                yield text

    inputs = (text for pair in itertools.zip_longest(*(
        itertools.islice(documents(paths[lang, 'train']), args.sample_documents)
        for lang in ('zh', 'en'))) for text in pair if text is not None)
    vocab, merges, audit = train_bpe_documents(inputs, args.vocab_size, SPECIAL_TOKENS)
    if len(vocab) != args.vocab_size:
        raise ValueError(f'Corpus yielded {len(vocab)} tokens. Supply more diverse TRAIN text.')
    tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    tp = root / 'artifacts/tokenizers' / args.name
    dst = root / 'datasets/pretrain' / args.name
    if tp.exists() or dst.exists():
        raise FileExistsError('Choose a new artifact name.')
    tp.mkdir(parents=True)
    dst.mkdir(parents=True)
    tokenizer.save(tp / 'tokenizer.json')
    sha = digest(tp / 'tokenizer.json')
    write_json(tp / 'manifest.json', dict(status='ready', vocab_size=len(vocab), tokenizer_sha256=sha, fit=audit))
    files = []
    for lang, split in splits:
        path = dst / f'{lang}-{split}.bin'
        count = 0
        with path.open('xb') as stream:
            for text in documents(paths[lang, split]):
                ids = tokenizer.encode(text) + [tokenizer.special_ids['<|endoftext|>']]
                stream.write(np.asarray(ids, dtype='<u2').tobytes())
                count += len(ids)
        if not count:
            raise ValueError(f'{lang}-{split} has no usable text.')
        files.append(dict(language=lang, split=split, path=path.relative_to(root).as_posix(),
                          tokens=count, sha256=digest(path)))
    write_json(dst / 'manifest.json', dict(status='ready', dtype='uint16', tokenizer=args.name,
                                          tokenizer_sha256=sha, files=files))
    return {'token_manifest': str(dst / 'manifest.json')}


def register_sft(root, args):
    import shutil

    tokenizer = root / 'artifacts/tokenizers' / args.tokenizer / 'tokenizer.json'
    sha = digest(tokenizer)
    splits = {'train': args.train, 'dev': args.dev}
    if args.test:
        splits['test'] = args.test
    families, files = {}, []
    dst = root / 'datasets/sft' / args.name
    for split, source in splits.items():
        count = 0
        for row in iter_jsonl(source):
            family = row['family_id']
            if not family or not row['source_ids'] or not clean_document(row['question']) or not clean_document(row['answer']):
                raise ValueError('Each QA record needs family_id, source_ids, question and answer.')
            if families.setdefault(family, split) != split:
                raise ValueError(f'Source family leaks across splits: {family}')
            count += 1
        if not count:
            raise ValueError(f'Empty {split} split.')
        files.append((split, source, count))
    dst.mkdir(parents=True, exist_ok=False)
    entries = []
    for split, source, count in files:
        target = dst / f'{split}.jsonl'
        shutil.copyfile(source, target)
        entries.append(dict(split=split, path=target.relative_to(root).as_posix(),
                            examples=count, sha256=digest(target)))
    write_json(dst / 'manifest.json', dict(status='ready', format='mlsystem-sft-v1',
                                          tokenizer=args.tokenizer, tokenizer_sha256=sha, files=entries))
    return {'sft_manifest': str(dst / 'manifest.json')}


def device_for(args):
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Select exactly one permitted CUDA device in CUDA_VISIBLE_DEVICES before starting.')
        return torch.device('cuda:0')
    return torch.device('cpu')


def train(root, args):
    import torch
    from mlsystem.config import ModelConfig
    from mlsystem.training import PretrainSettings, run_pretraining
    from mlsystem.sft import SFTSettings, run_sft

    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    device = device_for(args)
    output = root / 'artifacts/models' / args.name
    if args.bf16 and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('--bf16 requires a CUDA device with BF16 support.')
    if args.command == 'pretrain':
        return run_pretraining(root, output, ModelConfig(**cfg['model']), PretrainSettings(**cfg['training']),
                               root / cfg['token_manifest'], device=device, bf16=args.bf16)
    return run_sft(root, output, cfg['base_checkpoint'], root / cfg['sft_manifest'], SFTSettings(**cfg['training']),
                   device=device, bf16=args.bf16)


def generate(root, args):
    import torch
    from mlsystem.bpe import Tokenizer
    from mlsystem.data import conversation_prefix
    from mlsystem.inference import GenerationSettings, generate as generate_tokens
    from mlsystem.ols_rbf import RBFAdaptedModel
    from mlsystem.sft import load_base, load_sft

    device = device_for(args)
    path = root / 'artifacts/tokenizers' / args.tokenizer / 'tokenizer.json'
    tokenizer = Tokenizer.load(path)
    if args.kind == 'base':
        model, state, _ = load_base(root, args.checkpoint)
        model.to(device).eval()
    else:
        model, state = load_sft(root, args.checkpoint, device)
    if state['identity']['tokenizer_sha256'] != digest(path):
        raise ValueError('Checkpoint and selected tokenizer differ.')
    question_rows = ((row['question'], row.get('context', '')) for row in iter_jsonl(args.requests)) if args.requests else [(args.prompt, args.context)]
    online = None
    if args.backend == 'cuda-online':
        if device.type != 'cuda' or args.kind != 'sft':
            raise ValueError('cuda-online requires CUDA and an SFT checkpoint.')
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'inference_cuda'))
        from allarchitectures_inference_engine_v1 import make_engine
        from common_aux_v1_adapter import configure_model
        from online_cuda_engine_v2 import OnlineEngine

        route = state['settings']['route']
        hop = model.config.attention == 'hopfield_official_v8'
        if hop and model.config.d_model // model.config.num_heads != 64:
            raise ValueError('The bundled Hopfield CUDA kernel requires 64 coordinates per attention head.')
        if model.config.attention not in ('standard', 'hopfield_official_v8'):
            raise ValueError('Use the reference backend for legacy Hopfield checkpoints.')
        spec = dict(engine_suite='state-v1', advanced=hop, cache='fixed' if hop else 'static',
                    attention='manual' if hop else 'sdpa', precision='fp32' if route in ('genlora', 'ols') else 'fp16',
                    merged_reference=True, pack=True, shared_rope=True, last_query=route != 'ols',
                    target_prefill=True, target_buckets=True, joint_selection=True, integrated_selection=True,
                    gpu_feedback=route != 'ols', gpu_position=True, pinned_delivery=route != 'ols',
                    decode_attention='triton_direct' if hop else 'native_gqa')
        stops = {tokenizer.special_ids[s] for s in ('<|endoftext|>', '<|turn_end|>')}
        banned = [tokenizer.special_ids[s] for s in SPECIAL_TOKENS if tokenizer.special_ids[s] not in stops]
        spec['banned_token_ids'] = banned
        with torch.no_grad():
            configure_model(model, spec, root, tokenizer)
            online = OnlineEngine(make_engine(model, spec, []))
            online.setup()

    for question, context in question_rows:
        prompt = question if args.kind == 'base' else conversation_prefix(question, context)
        if isinstance(model, RBFAdaptedModel) and online is None and hasattr(model.adapter, 'encode_question'):
            model.adapter._question_mean = model.adapter.encode_question(tokenizer, question)
        if online is None:
            result = generate_tokens(model, tokenizer, prompt, GenerationSettings(
                max_new_tokens=args.max_new_tokens, temperature=0, top_p=1., repetition_penalty=1.))
        else:
            ids = tokenizer.encode(prompt)
            if not ids or len(ids) + args.max_new_tokens > online.capacity:
                raise ValueError('Prompt and generation budget exceed the online context.')
            with torch.no_grad():
                online.before_request(len(ids), args.max_new_tokens)
                result = online.generate(tokenizer, {'prompt_ids': ids, 'question': question},
                                         {'max_new_tokens': args.max_new_tokens}, banned, stops)
                online.after_request()
            result['text'] = tokenizer.decode(result['token_ids'])
        print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--name', required=True, type=name)
    p.add_argument('--vocab-size', type=int, choices=range(512, 65536), metavar='512..65535', default=512)
    p.add_argument('--sample-documents', type=int, default=3000)
    p = sub.add_parser('register-sft')
    p.add_argument('--tokenizer', required=True, type=name)
    p.add_argument('--name', required=True, type=name)
    for split in ('train', 'dev', 'test'):
        p.add_argument('--' + split, required=split != 'test', type=Path)
    for cmd in ('pretrain', 'sft'):
        p = sub.add_parser(cmd)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--name', required=True, type=name)
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        p.add_argument('--bf16', action='store_true')
    p = sub.add_parser('generate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True, type=name)
    p.add_argument('--kind', choices=('base', 'sft'), default='sft')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--backend', choices=('reference', 'cuda-online'), default='reference')
    query = p.add_mutually_exclusive_group(required=True)
    query.add_argument('--prompt')
    query.add_argument('--requests', type=Path)
    p.add_argument('--context', default='')
    p.add_argument('--max-new-tokens', type=int, default=16)
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.command == 'prepare':
        result = prepare(root, args)
    elif args.command == 'register-sft':
        result = register_sft(root, args)
    elif args.command in ('pretrain', 'sft'):
        result = train(root, args)
    else:
        if not 1 <= args.max_new_tokens <= 512:
            parser.error('--max-new-tokens must be in [1, 512]')
        generate(root, args)
        return
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

