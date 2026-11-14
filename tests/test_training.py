import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mlsystem.config import ModelConfig
from mlsystem.training import PretrainSettings,run_pretraining,file_sha


class TrainingTests(unittest.TestCase):
    def test_real_updates_and_exact_resume_without_data_rng_drift(self):
        torch.set_num_threads(2)
        base=ROOT/'tmp/core-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary);data=root/'datasets/pretrain/fixture';data.mkdir(parents=True)
            tokenizer=root/'artifacts/tokenizers/fixture';tokenizer.mkdir(parents=True)
            (tokenizer/'tokenizer.json').write_text('{}',encoding='utf-8')
            tokenizer_sha=file_sha(tokenizer/'tokenizer.json')
            (tokenizer/'manifest.json').write_text(json.dumps({'status':'ready','vocab_size':512,
                                                               'tokenizer_sha256':tokenizer_sha}),encoding='utf-8')
            manifest={'status':'ready','dtype':'uint16','tokenizer':'fixture','tokenizer_sha256':tokenizer_sha,'files':[]}
            for lang in ['zh','en']:
                for split in ['train','dev']:
                    p=data/f'{lang}-{split}.bin'
                    np.full(256,3,dtype='<u2').tofile(p)
                    manifest['files'].append({'path':p.relative_to(root).as_posix(),'language':lang,
                                             'split':split,'tokens':256,'sha256':file_sha(p)})
            mp=data/'manifest.json';mp.write_text(json.dumps(manifest),encoding='utf-8')
            cfg=ModelConfig(vocab_size=512,context_length=32,d_model=64,num_layers=1,num_heads=4,num_kv_heads=2,d_ff=192)
            settings=PretrainSettings(steps=4,microbatch=2,accumulation=1,sequence_length=32,
                                      learning_rate=.004,min_learning_rate=.001,warmup_steps=0,
                                      eval_every=2,eval_batches=1,save_every=2)
            full=root/'artifacts/models/continuous'
            result=run_pretraining(root,full,cfg,settings,mp,device=torch.device('cpu'))
            self.assertEqual(result['status'],'completed')
            rows=[json.loads(x) for x in (full/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertLess(rows[-1]['ce'],rows[0]['ce'])
            self.assertEqual(result['tokens_seen'],256)
            calls=0
            def stop():
                nonlocal calls
                calls+=1
                return calls>2
            divided=root/'artifacts/models/resumed'
            paused=run_pretraining(root,divided,cfg,settings,mp,device=torch.device('cpu'),stop_requested=stop)
            self.assertEqual(paused['status'],'cancelled')
            self.assertEqual(paused['step'],2)
            resumed=run_pretraining(root,divided,cfg,settings,mp,device=torch.device('cpu'),resume=True)
            self.assertEqual(resumed['status'],'completed')
            a=torch.load(full/'last.pt',weights_only=True,map_location='cpu')
            b=torch.load(divided/'last.pt',weights_only=True,map_location='cpu')
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            self.assertEqual(a['sampler_state'],b['sampler_state'])


if __name__=='__main__':unittest.main()
