import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mlsystem.bpe import Tokenizer, train_bpe_documents
import random
from mlsystem.data import SPECIAL_TOKENS, document_split, clean_document, encode_sft


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab={i:bytes([i]) for i in range(256)}
        for s in SPECIAL_TOKENS: vocab[len(vocab)]=s.encode('utf-8')
        cls.tok=Tokenizer(vocab,[],SPECIAL_TOKENS)

    def test_answer_shift_mask_and_terminator(self):
        record={'question':'What is VLA?','answer':'视觉、语言、动作。','context':'Source text.'}
        x,y=encode_sft(record,self.tok,512)
        targets=[v for v in y if v!=-100]
        self.assertEqual(targets,self.tok.encode(record['answer'])+[self.tok.special_ids['<|turn_end|>']])
        first=next(i for i,v in enumerate(y) if v!=-100)
        self.assertEqual(x[first],self.tok.special_ids['<|assistant|>'])
        self.assertEqual(len(x),len(y))
        self.assertEqual(len(x),512)

    def test_unicode_roundtrip_and_bad_input(self):
        text='中文与 English\n科学🚀 café'
        self.assertEqual(self.tok.decode(self.tok.encode(text)),text)
        with self.assertRaises(ValueError): clean_document('\udcef')
        with self.assertRaises(ValueError): clean_document('hi <|assistant|>')
        with self.assertRaises(ValueError): encode_sft({'question':'x'*400,'answer':'y'},self.tok,32)

    def test_family_splits_stable_and_separated(self):
        ids=[f'Q{i}' for i in range(1000)]
        a={s:{i for i in ids if document_split(i)==s} for s in ['train','dev','test']}
        self.assertTrue(all(a.values()))
        self.assertFalse(a['train']&a['dev'] or a['train']&a['test'] or a['dev']&a['test'])
        self.assertEqual(document_split('Q10'),document_split('Q10'))

    def test_bounded_bpe_uses_actual_documents(self):
        vocab,merges,audit=train_bpe_documents(['中文 English.']*8,280,SPECIAL_TOKENS,max_bytes=60)
        tokenizer=Tokenizer(vocab,merges,SPECIAL_TOKENS)
        self.assertLessEqual(audit['sample_bytes'],60)
        self.assertEqual(tokenizer.decode(tokenizer.encode('中文 English.')),'中文 English.')

    def test_heap_bpe_matches_rank_order_reference_for_long_pieces(self):
        texts=['神经网络学习语言模型与机器人的知识。'*30,'mississippi'*60,'abcababcabc'*70]
        vocab,merges,_=train_bpe_documents(texts,700,SPECIAL_TOKENS,max_bytes=20000)
        tokenizer=Tokenizer(vocab,merges,SPECIAL_TOKENS)
        rng=random.Random(11)
        pieces=[text.encode('utf-8') for text in texts]
        pieces += [bytes(rng.randrange(256) for _ in range(300)) for _ in range(30)]
        pieces += [(''.join(rng.choice('神经网络学习中文') for _ in range(100))).encode('utf-8') for _ in range(30)]
        for piece in pieces:
            self.assertEqual(tokenizer._merge_piece(piece),tokenizer._merge_piece_slow(piece))


if __name__=='__main__': unittest.main()
