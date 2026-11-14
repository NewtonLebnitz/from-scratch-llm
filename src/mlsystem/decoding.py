"""Inference-only incremental GQA/Hopfield decoding; training modules stay frozen.

K/V tensors persist across tokens of ONE request. Hopfield score state persists
across layers for the current query rows, never across unrelated requests.
GenLoRA A/B are evaluated once per request with fixed inference parameters.
"""
import math
import torch
from .components import softmax, silu
from .genlora import AdapterLinear
from .model import LanguageModel
from .ols_rbf import RBFAdaptedModel


class CachedDecoder:
    def __init__(self, model):
        if model.training:raise ValueError('Cached decoding requires eval mode')
        self.model=model
        self.base=model.base if isinstance(model,RBFAdaptedModel) else model
        if not isinstance(self.base,LanguageModel):raise TypeError('Unsupported model for incremental decoding')
        self.kv=[None]*len(self.base.layers)
        self.matrices={};self.position=0;self.batch=None

    def linear(self, module, x):
        if not isinstance(module,AdapterLinear):return module(x)
        key=id(module)
        if key not in self.matrices:
            a,b=module.matrices();self.matrices[key]=(a.detach(),b.detach())
        a,b=self.matrices[key]
        return module.base_layer(x)+((x@a.T)@b.T)*module.scale

    @torch.no_grad()
    def last_logits(self, ids):
        if ids.ndim!=2 or ids.shape[1]<1:raise ValueError('Need nonempty [batch,tokens]')
        if self.batch is not None and ids.shape[0]!=self.batch:raise ValueError('Cache batch size changed')
        if self.position+ids.shape[1]>self.base.context_length:raise ValueError('Cache exceeds context')
        self.batch=ids.shape[0];batch,length=ids.shape
        positions=torch.arange(self.position,self.position+length,device=ids.device)
        keys=torch.arange(self.position+length,device=ids.device)
        mask=keys[None,:]<=positions[:,None]
        x=self.base.token_embeddings(ids)
        state=None
        for index,layer in enumerate(self.base.layers):
            normalized=layer.ln1(x);att=layer.attn
            q=self.linear(att.q_proj,normalized).reshape(batch,length,att.num_heads,att.head_dim).transpose(1,2)
            k=self.linear(att.k_proj,normalized).reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
            v=self.linear(att.v_proj,normalized).reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
            q,k=att.rope(q,positions),att.rope(k,positions)
            if self.kv[index] is not None:
                old_k,old_v=self.kv[index];k=torch.cat((old_k,k),dim=-2);v=torch.cat((old_v,v),dim=-2)
            self.kv[index]=(k,v)
            groups=att.num_heads//att.num_kv_heads
            grouped_k=k.repeat_interleave(groups,dim=1);grouped_v=v.repeat_interleave(groups,dim=1)
            scores=(q@grouped_k.transpose(-1,-2)).float()/math.sqrt(att.head_dim)
            scores=scores.masked_fill(~mask,0)
            if att.mode in ['hopfield','hopfield_official_v8']:
                state=(scores if att.mode=='hopfield_official_v8' else (1-att.alpha_prime)*scores) if state is None else att.alpha_prime*state+(1-att.alpha_prime)*scores
                logits=state
            else:logits=scores
            probabilities=softmax(logits.masked_fill(~mask,float('-inf')),-1)
            attended=(probabilities.to(v.dtype)@grouped_v).transpose(1,2).reshape(batch,length,-1)
            if att.mode=='hopfield_official_v8':attended=(1-att.alpha)*attended+att.alpha*normalized
            update=self.linear(att.output_proj,attended)
            if att.mode=='hopfield':update=att.alpha*normalized+(1-att.alpha)*update
            x=x+update
            normalized=layer.ln2(x);ff=layer.ffn
            x=x+self.linear(ff.w2,silu(self.linear(ff.w1,normalized))*self.linear(ff.w3,normalized))
        h=self.base.ln_final(x[:,-1])
        result=self.linear(self.base.lm_head,h)
        if isinstance(self.model,RBFAdaptedModel):result=result.float()+self.model.adapter(h)
        self.position+=length
        return result


def make_decoder(model):
    base=model.base if isinstance(model,RBFAdaptedModel) else model
    return CachedDecoder(model) if isinstance(base,LanguageModel) else None
