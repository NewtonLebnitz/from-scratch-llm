"""Controlled pretraining engine: bilingual sampling, checkpoints, cancellation and metrics.

Import this module only after the GPU7 guard in production. CPU callers are for
small correctness checks; this is not a background service or an alternate GPU path.
"""
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import torch
from .model import LanguageModel
from .config import ModelConfig
from .optim import AdamW,cross_entropy,gradient_clipping,get_lr_cosine_schedule


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix(path.suffix+'.writing')
    with part.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(part,path)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class PretrainSettings:
    steps:int=1000
    microbatch:int=4
    accumulation:int=4
    sequence_length:int=512
    learning_rate:float=0.0003
    min_learning_rate:float=0.00003
    warmup_steps:int=100
    weight_decay:float=0.1
    grad_clip:float=1.0
    chinese_fraction:float=0.5
    eval_every:int=250
    eval_batches:int=8
    save_every:int=1000
    seed:int=7

    def validate(self,model):
        bounds={'steps':(1,500000),'microbatch':(1,16),'accumulation':(1,64),
                'sequence_length':(32,model.context_length),'warmup_steps':(0,self.steps),
                'eval_every':(1,100000),'eval_batches':(1,64),'save_every':(1,100000)}
        for key,(lo,hi) in bounds.items():
            x=getattr(self,key)
            if type(x)is not int or not lo<=x<=hi:raise ValueError(f'{key} must be integer in [{lo},{hi}]')
        if not 0<self.min_learning_rate<=self.learning_rate<=.01:raise ValueError('Invalid learning rate range')
        if not 0<=self.weight_decay<=1 or not 0<self.grad_clip<=100:raise ValueError('Invalid decay/clip')
        if not .1<=self.chinese_fraction<=.9:raise ValueError('Both languages must get at least 10% sampling weight')
        return self


class BilingualTokens:
    def __init__(self,root,manifest_path,context,seed):
        self.root=Path(root).resolve();self.manifest_path=Path(manifest_path)
        manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status')!='ready' or manifest.get('dtype')!='uint16':
            raise ValueError('Need a ready versioned uint16 token corpus')
        self.identity=file_sha(manifest_path)
        self.tokenizer_sha256=manifest['tokenizer_sha256']
        name=manifest['tokenizer']
        if not isinstance(name,str) or not name.replace('-','').replace('_','').isalnum():
            raise ValueError('Invalid registered tokenizer name')
        tokenizer_path=self.root/'artifacts/tokenizers'/name
        tokenizer_manifest=json.loads((tokenizer_path/'manifest.json').read_text(encoding='utf-8'))
        if tokenizer_manifest['status']!='ready' or file_sha(tokenizer_path/'tokenizer.json')!=self.tokenizer_sha256:
            raise ValueError('Token dataset and actual tokenizer disagree')
        self.vocab_size=tokenizer_manifest['vocab_size']
        self.rng=np.random.default_rng(seed);self.context=context;self.arrays={}
        for row in manifest['files']:
            path=(self.root/row['path']).resolve()
            if not path.is_relative_to(self.root/'datasets/pretrain') or file_sha(path)!=row['sha256']:
                raise ValueError('Token source path/hash mismatch')
            if path.stat().st_size!=row['tokens']*2 or row['tokens']<=context:
                raise ValueError('Token source too short or incorrectly sized')
            self.arrays[(row['language'],row['split'])]=np.memmap(path,dtype='<u2',mode='r')
        if any((lang,split)not in self.arrays for lang in ['zh','en'] for split in ['train','dev']):
            raise ValueError('Both languages need train and dev data')

    def batch(self,split,batch_size,chinese_fraction,device):
        values=[]
        for _ in range(batch_size):
            lang='zh' if self.rng.random()<chinese_fraction else 'en'
            a=self.arrays[(lang,split)]
            start=int(self.rng.integers(0,len(a)-self.context))
            values.append(np.asarray(a[start:start+self.context+1],dtype=np.int64))
        tensor=torch.from_numpy(np.stack(values)).to(device)
        return tensor[:,:-1],tensor[:,1:]


def save_training_checkpoint(path,model,optimizer,step,sampler,identity,settings,best_dev):
    path=Path(path)
    state={'format':'mlsystem-pretrain-v1','model_config':model.config.dictionary(),
           'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
           'settings':asdict(settings),'identity':identity,'best_dev':best_dev,
           'sampler_state':sampler.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
           'python_rng':random.getstate(),
           'cuda_rng':torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None}
    part=path.with_suffix('.writing')
    with part.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(part,path)


@torch.no_grad()
def evaluate(model,sampler,settings,device,bf16):
    # Evaluation has an independent deterministic RNG and never consumes training RNG.
    previous=sampler.rng.bit_generator.state
    sampler.rng=np.random.default_rng(settings.seed+999)
    was_training=model.training;model.eval();losses={}
    try:
        for lang,weight in [('zh',1.0),('en',0.0)]:
            total=0.
            for _ in range(settings.eval_batches):
                x,y=sampler.batch('dev',settings.microbatch,weight,device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                    loss=cross_entropy(model(x),y)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
                total+=float(loss)
            value=total/settings.eval_batches
            losses[lang]={'ce':value,'ppl':math.exp(min(value,50))}
        losses['balanced_ce']=.5*(losses['zh']['ce']+losses['en']['ce'])
        return losses
    finally:
        sampler.rng=np.random.default_rng()
        sampler.rng.bit_generator.state=previous
        model.train(was_training)


def run_pretraining(root,output,config,settings,token_manifest,*,device,resume=False,stop_requested=lambda:False,bf16=False,purpose='training'):
    root,output=Path(root).resolve(),Path(output).resolve()
    if not output.is_relative_to(root/'artifacts/models') or output==root/'artifacts/models':
        raise ValueError('Model output must be a dedicated directory under artifacts/models')
    config.validate();settings.validate(config)
    if purpose not in ['training','hardware_validation']:raise ValueError('Invalid artifact purpose')
    if device.type=='cuda' and (device.index not in [None,0] or torch.cuda.device_count()!=1):
        raise RuntimeError('Production GPU selection must be the single UUID-bound visible device')
    if resume:
        if not (output/'last.pt').is_file():raise ValueError('No resumable checkpoint')
    data=BilingualTokens(root,token_manifest,settings.sequence_length,settings.seed)
    if config.vocab_size!=data.vocab_size:raise ValueError('Model vocabulary must match actual trained BPE vocabulary')
    if not resume:output.mkdir(parents=True,exist_ok=False)
    code=hashlib.sha256()
    for filename in ['components.py','config.py','model.py','optim.py','training.py']:
        code.update((Path(__file__).parent/filename).read_bytes())
    identity={'data_manifest_sha256':data.identity,'tokenizer_sha256':data.tokenizer_sha256,
              'training_code_sha256':code.hexdigest()}
    random.seed(settings.seed);torch.manual_seed(settings.seed)
    model=LanguageModel(config).to(device)
    decay=[p for p in model.parameters() if p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.ndim<2]
    optimizer=AdamW([{'params':decay,'weight_decay':settings.weight_decay},
                     {'params':no_decay,'weight_decay':0.}],lr=settings.learning_rate,betas=(.9,.95))
    start=0;best_dev=math.inf
    if resume:
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if state['identity']!=identity or state['settings']!=asdict(settings) or state['model_config']!=config.dictionary():
            raise ValueError('Resume data/tokenizer/settings/architecture differ')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        data.rng.bit_generator.state=state['sampler_state'];torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'])
        start=state['step'];best_dev=state['best_dev'];del state
    model.train();step=start;seen=start*settings.microbatch*settings.accumulation*settings.sequence_length
    status={'status':'running','stage':'pretrain','purpose':purpose,'attention':config.attention,'pid':os.getpid(),
            'model_config':config.dictionary(),'settings':asdict(settings),'identity':identity,
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'step':step,'tokens_seen':seen,'parent':None}
    atomic_json(output/'status.json',status)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    try:
        for step in range(start+1,settings.steps+1):
            if stop_requested():step-=1;break
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            lr=get_lr_cosine_schedule(step,settings.learning_rate,settings.min_learning_rate,settings.warmup_steps,settings.steps)
            for group in optimizer.param_groups:group['lr']=lr
            loss_value=0.
            for _ in range(settings.accumulation):
                x,y=data.batch('train',settings.microbatch,settings.chinese_fraction,device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                    loss=cross_entropy(model(x),y)/settings.accumulation
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss; checkpoint not overwritten')
                loss.backward();loss_value+=float(loss.detach())
            norm=gradient_clipping(model.parameters(),settings.grad_clip)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient; checkpoint not overwritten')
            optimizer.step()
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-tick
            tokens=settings.microbatch*settings.accumulation*settings.sequence_length;seen+=tokens
            row={'step':step,'tokens_seen':seen,'ce':loss_value,'lr':lr,'grad_norm':float(norm),
                 'update_seconds':seconds,'tokens_per_second':tokens/seconds,
                 'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None}
            if step%settings.eval_every==0 or step==settings.steps:
                row['dev']=evaluate(model,data,settings,device,bf16)
                if row['dev']['balanced_ce']<best_dev:
                    best_dev=row['dev']['balanced_ce']
                    save_training_checkpoint(output/'best.pt',model,optimizer,step,data,identity,settings,best_dev)
            with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            status.update(step=step,tokens_seen=seen,last=row,best_dev=best_dev if math.isfinite(best_dev) else None)
            atomic_json(output/'status.json',status)
            if step%settings.save_every==0 or step==settings.steps:
                save_training_checkpoint(output/'last.pt',model,optimizer,step,data,identity,settings,best_dev)
        save_training_checkpoint(output/'last.pt',model,optimizer,step,data,identity,settings,best_dev)
        status.update(status='completed' if step==settings.steps else 'cancelled',step=step,
                      tokens_seen=seen,session_seconds=time.perf_counter()-started)
        if step==settings.steps:
            status['checkpoint_sha256']=file_sha(output/'best.pt')
            status['checkpoint']='best.pt'
    except BaseException as exc:
        status.update(status='failed',error=type(exc).__name__+': '+str(exc),step=step)
        raise
    finally:atomic_json(output/'status.json',status)
    return status
