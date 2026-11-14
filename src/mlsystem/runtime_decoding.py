"""Inference-only FP32 packed projections and GenLoRA reparameterization; original model tensors are never modified.
Original request-local KV/GenLoRA matrix cache already existed, not claimed as new.
"""
import math
import torch
from mlsystem.components import softmax, silu
from mlsystem.genlora import AdapterLinear
from mlsystem.model import LanguageModel
from mlsystem.ols_rbf import RBFAdaptedModel


def parameter_signature(base):
    return tuple((id(p),p._version,p.device,p.dtype)for p in base.parameters())

MAX_DERIVED_BYTES=768*1024**2

def derived_bytes(base):
    total=sum(m.base_layer.weight.numel()*m.base_layer.weight.element_size()for m in base.modules()if isinstance(m,AdapterLinear))
    for layer in base.layers:
        for modules in [(layer.attn.q_proj,layer.attn.k_proj,layer.attn.v_proj),(layer.ffn.w1,layer.ffn.w3)]:
            total+=sum(m.weight.numel()*m.weight.element_size()for m in modules if not isinstance(m,AdapterLinear))
    return total

class RuntimePlan:
    def __init__(self,base):
        from mlsystem.components import Linear
        if not 0<derived_bytes(base)<=MAX_DERIVED_BYTES:raise ValueError('Derived inference weight budget exceeded')
        self.signature=parameter_signature(base);self.effective={};self.fused={};self.variant='fusion'
        for module in base.modules():
            if isinstance(module,AdapterLinear):
                a,b=module.matrices();weight=(module.base_layer.weight+(b@a)*module.scale).detach()
            elif isinstance(module,Linear):weight=module.weight.detach()
            else:continue
            self.effective[id(module)]=weight
        for layer in base.layers:
            for key,modules in [(id(layer.attn),(layer.attn.q_proj,layer.attn.k_proj,layer.attn.v_proj)),(id(layer.ffn),(layer.ffn.w1,layer.ffn.w3))]:
                sizes=[self.effective[id(m)].shape[0]for m in modules]
                self.fused[key]=torch.cat([self.effective[id(m)]for m in modules],0)
                for module,weight in zip(modules,self.fused[key].split(sizes,0)):self.effective[id(module)]=weight
        first=base.layers[0].attn.rope
        self.shared_rope=all(torch.equal(first.cos,l.attn.rope.cos)and torch.equal(first.sin,l.attn.rope.sin)for l in base.layers)
        self.cache_bytes=derived_bytes(base)

def runtime_plan(base):
    plan=getattr(base,'_runtime_inference_plan_v4',None)
    if plan is None or plan.signature!=parameter_signature(base):
        plan=RuntimePlan(base);base._runtime_inference_plan_v4=plan
    return plan

def rotate(x,c,s):
    even,odd=x[...,0::2].float(),x[...,1::2].float()
    return torch.stack((even*c-odd*s,even*s+odd*c),-1).flatten(-2).to(x.dtype)

class MergedPlanDecoder:
    def __init__(self, model):
        if model.training:raise ValueError('Cached decoding requires eval mode')
        self.model=model
        self.backend_label='FP32 packed QKV/SwiGLU / shared RoPE / final-block last-query; static KV; Hopfield EMA preserved'
        self.base=model.base if isinstance(model,RBFAdaptedModel) else model
        if not isinstance(self.base,LanguageModel):raise TypeError('Unsupported model for incremental decoding')
        self.kv=[None]*len(self.base.layers)
        previous=getattr(self.base,'_runtime_inference_plan_v4',None)
        self.plan=runtime_plan(self.base);self.plan_reused=self.plan is previous
        self.position=0;self.batch=None

    def linear(self,module,x):
        return x@self.plan.effective[id(module)].T

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
        cos=sin=None
        if self.plan.shared_rope:
            cos=self.base.layers[0].attn.rope.cos[positions][None,None]
            sin=self.base.layers[0].attn.rope.sin[positions][None,None]
        for index,layer in enumerate(self.base.layers):
            normalized=layer.ln1(x);att=layer.attn
            last_only=index==len(self.base.layers)-1 and length>1
            query=normalized[:,-1:] if last_only else normalized
            query_length=query.shape[1]
            query_mask=mask[-1:] if last_only else mask
            if self.plan.variant=='fusion' and not last_only:
                nq=att.num_heads*att.head_dim;nk=att.num_kv_heads*att.head_dim
                q,k,v=(normalized@self.plan.fused[id(att)].T).split([nq,nk,nk],-1)
                q=q.reshape(batch,query_length,att.num_heads,att.head_dim).transpose(1,2)
                k=k.reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
                v=v.reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
            else:
                q=self.linear(att.q_proj,query).reshape(batch,query_length,att.num_heads,att.head_dim).transpose(1,2)
                k=self.linear(att.k_proj,normalized).reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
                v=self.linear(att.v_proj,normalized).reshape(batch,length,att.num_kv_heads,att.head_dim).transpose(1,2)
            if cos is None:
                q,k=att.rope(q,positions[-1:] if last_only else positions),att.rope(k,positions)
            else:
                q,k=rotate(q,cos[:,:,-query_length:],sin[:,:,-query_length:]),rotate(k,cos,sin)
            if last_only:
                x=x[:,-1:];normalized=normalized[:,-1:]
                if state is not None:state=state[:,:,-1:,:]
            if self.kv[index] is None:
                shape=(batch,att.num_kv_heads,self.base.context_length,att.head_dim)
                self.kv[index]=(torch.empty(shape,device=k.device,dtype=k.dtype),torch.empty(shape,device=v.device,dtype=v.dtype))
            old_k,old_v=self.kv[index];end=self.position+length
            old_k[:,:,self.position:end].copy_(k);old_v[:,:,self.position:end].copy_(v)
            k=old_k[:,:,:end];v=old_v[:,:,:end]
            groups=att.num_heads//att.num_kv_heads
            grouped_k=k.repeat_interleave(groups,dim=1);grouped_v=v.repeat_interleave(groups,dim=1)
            if att.mode=='standard':
                attended=torch.nn.functional.scaled_dot_product_attention(q,grouped_k,grouped_v,attn_mask=query_mask[None,None] if length>1 else None,dropout_p=0.,is_causal=False).transpose(1,2).reshape(batch,query_length,-1)
            else:
                # Hopfield retains cross-layer score dynamics; never substituted by plain SDPA.
                scores=(q@grouped_k.transpose(-1,-2)).float()/math.sqrt(att.head_dim)
                scores=scores.masked_fill(~query_mask,0)
                state=(scores if att.mode=='hopfield_official_v8' else (1-att.alpha_prime)*scores) if state is None else att.alpha_prime*state+(1-att.alpha_prime)*scores
                probabilities=softmax(state.masked_fill(~query_mask,float('-inf')),-1)
                attended=(probabilities.to(v.dtype)@grouped_v).transpose(1,2).reshape(batch,query_length,-1)
            if att.mode=='hopfield_official_v8':attended=(1-att.alpha)*attended+att.alpha*normalized
            update=self.linear(att.output_proj,attended)
            if att.mode=='hopfield':update=att.alpha*normalized+(1-att.alpha)*update
            x=x+update
            normalized=layer.ln2(x);ff=layer.ffn
            if self.plan.variant=='fusion':
                gate,value=(normalized@self.plan.fused[id(ff)].T).chunk(2,-1)
                x=x+self.linear(ff.w2,torch.nn.functional.silu(gate)*value)
            else:x=x+self.linear(ff.w2,silu(self.linear(ff.w1,normalized))*self.linear(ff.w3,normalized))
        h=self.base.ln_final(x[:,-1])
        result=self.linear(self.base.lm_head,h)
        if isinstance(self.model,RBFAdaptedModel):result=result.float()+self.model.adapter(h)
        self.position+=length
        return result



def make_decoder(model):
    base=model.base if isinstance(model,RBFAdaptedModel) else model
    if not isinstance(base,LanguageModel):return None
    first=next(base.parameters())
    if first.device.type=='cpu' and first.dtype==torch.float32 and 0<derived_bytes(base)<=MAX_DERIVED_BYTES:
        if any(isinstance(m,AdapterLinear) for m in base.modules()):
            from .runtime_reference_v3 import make_decoder as genlora_decoder
            decoder=genlora_decoder(model);decoder.backend_label='GenLoRA FP32 effective weights v3 / shared RoPE / last-query; SSE transport'
            return decoder
        return MergedPlanDecoder(model)
    from .fast_decoding import make_decoder as stable_decoder
    return stable_decoder(model)
