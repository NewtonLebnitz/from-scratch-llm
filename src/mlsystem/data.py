"""Document-level split identities and real autoregressive full-answer SFT batches."""
import hashlib
import json
import re
import unicodedata

SPECIAL_TOKENS = ['<|endoftext|>', '<|system|>', '<|user|>', '<|assistant|>', '<|turn_end|>', '<|pad|>']


def clean_document(text):
    if not isinstance(text,str):
        raise ValueError('Document text must be a string')
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError('Document contains invalid Unicode surrogate characters') from exc
    text = unicodedata.normalize('NFC',text).replace('\r\n','\n').replace('\r','\n')
    text = ''.join(c for c in text if c in '\n\t' or not unicodedata.category(c).startswith('C'))
    text = re.sub(r'[ \t]+',' ',text)
    text = re.sub(r'\n{4,}','\n\n\n',text).strip()
    if any(marker in text for marker in SPECIAL_TOKENS):
        raise ValueError('Source contains reserved conversation markers')
    return text


def document_split(family_id, seed=7):
    """Stable source-family split; translated/alternate versions share the family ID."""
    value = int(hashlib.sha256(f'{seed}:{family_id}'.encode('utf-8')).hexdigest()[:8],16) % 10000
    return 'test' if value < 150 else 'dev' if value < 300 else 'train'


def content_hash(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def conversation_prefix(question, context='', system='Answer clearly. Use the supplied sources when provided. If the sources do not support an answer, say what is missing.'):
    question = clean_document(question)
    context = clean_document(context)
    system = clean_document(system)
    if not question:
        raise ValueError('Question is empty')
    message = ('Sources:\n'+context+'\n\nQuestion:\n' if context else '') + question
    return f'<|system|>{system}<|turn_end|><|user|>{message}<|turn_end|><|assistant|>'


def encode_sft(record, tokenizer, context_length):
    """input[t] predicts target[t]; only answer/turn-end tokens contribute to CE.

    Overlength examples are rejected, not silently cropped into unrelated questions.
    The data preparation stage must shorten source context with provenance.
    """
    prefix = conversation_prefix(record['question'], record.get('context',''), record.get('system',
        'Answer clearly. Use the supplied sources when provided. If the sources do not support an answer, say what is missing.'))
    answer = clean_document(record['answer'])
    if not answer:
        raise ValueError('Training answer must not be empty')
    before = tokenizer.encode(prefix)
    after = tokenizer.encode(answer) + [tokenizer.special_ids['<|turn_end|>']]
    sequence = before + after
    if len(sequence)-1 > context_length:
        raise ValueError('SFT example exceeds model context; prepare a shorter supported example')
    inputs = sequence[:-1]
    labels = [-100] * (len(before)-1) + after
    pad = context_length-len(inputs)
    return inputs+[tokenizer.special_ids['<|pad|>']]*pad, labels+[-100]*pad


def iter_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'Invalid JSONL {path}:{number}') from exc
