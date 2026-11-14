"""Decode-only fused kernel; preserve frozen v2/v3 and failed dynamic-prefill evidence."""
import math,torch
from torch.nn import functional as F
from mlsystem.components import silu,softmax
from hopfield_inference_engine_v3 import HopfieldEngine as EagerContractFixed
from hopfield_inference_engine_v2 import make_engine as frozen_factory,load_model

class HopfieldEngine(EagerContractFixed):
 def phase_attention_with_state(self,q,k,v,mask,state,a,ks=None,vs=None,*,fixed=False):
  # The fused single-query backend is a decode backend. Dynamic prefill uses
  # the frozen original attention, including the last-block single-query case.
  if not fixed and self.spec.get('decode_attention')=='triton_direct':
   return self.prefill_attention_with_state(q,k,v,mask,state,a,ks,vs)
  return self.attention_with_state(q,k,v,mask,state,a,ks,vs)

 def forward(self,ids,positions,*,fixed=False):
   b,n=ids.shape;x=self.base.token_embeddings(ids);state=None
   if self.cache=='none':keys=positions
   elif fixed:keys=self.arange[:self.decode_bucket]
   else:keys=torch.arange(self.position+n,device=ids.device)
   mask=keys[None,:]<=positions[:,None];shared=self.spec.get('shared_rope',False)
   if shared:
    rope=self.base.layers[0].attn.rope;c=rope.cos[positions][None,None];s=rope.sin[positions][None,None]
   for i,layer in enumerate(self.base.layers):
    a=layer.attn;f=layer.ffn;y=layer.ln1(x)
    last=self.spec.get('last_query',False)and i==len(self.base.layers)-1 and n>1 and not self.spec.get('all_logits')
    nq=1 if last else n;query=y[:,-1:]if last else y;qp=positions[-1:]if last else positions
    if self.spec.get('pack'):
     wqkv,wkv,wffn=self.packed[i];dq=a.num_heads*a.head_dim;dk=a.num_kv_heads*a.head_dim
     if last:q=self.linear(a.q_proj,query);k,v=(y@wkv.T).split([dk,dk],-1)
     else:q,k,v=(y@wqkv.T).split([dq,dk,dk],-1)
    else:q=self.linear(a.q_proj,query);k=self.linear(a.k_proj,y);v=self.linear(a.v_proj,y)
    q=q.reshape(b,nq,a.num_heads,a.head_dim).transpose(1,2);k=k.reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2);v=v.reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
    if shared:q=self.rotate(q,c[:,:,-nq:],s[:,:,-nq:]);k=self.rotate(k,c,s)
    else:q,k=a.rope(q,qp),a.rope(k,positions)
    if last:
     x=x[:,-1:];query=query[:,-1:]
     if state is not None:state=state[:,:,-1:,:]
    km=vm=None
    if self.cache!='none':
     if self.cache=='dynamic':
      if self.position:k=torch.cat((self.kv[i][0],k),2);v=torch.cat((self.kv[i][1],v),2)
      self.kv[i]=(k,v)
     else:
      kc,vc=self.kv[i];write=positions if self.table is None else self.table[positions//16]*16+positions%16
      if self.kv_scales:
       sk=k.abs().amax(-1).float().clamp_min(1e-12)/127;sv=v.abs().amax(-1).float().clamp_min(1e-12)/127
       kc.index_copy_(2,write,(k/sk[...,None]).round().clamp(-127,127).to(torch.int8));vc.index_copy_(2,write,(v/sv[...,None]).round().clamp(-127,127).to(torch.int8))
       ks,vs=self.kv_scales[i];ks.index_copy_(2,write,sk);vs.index_copy_(2,write,sv)
       if self.position:
        k,v=kc,vc;km,vm=ks,vs
      else:
       kc.index_copy_(2,write,k);vc.index_copy_(2,write,v)
       if self.table is not None:
        if nq==1:k,v=kc,vc
        else:
         read=self.table[keys//16]*16+keys%16;k,v=kc.index_select(2,read),vc.index_select(2,read)
       else:k,v=kc[:,:,:len(keys)],vc[:,:,:len(keys)]
    att,state=self.phase_attention_with_state(q,k,v,mask[-1:]if last else mask,state,a,km,vm,fixed=fixed)
    att=att.transpose(1,2).reshape(b,nq,-1)
    # Official v8: visible-state mixture precedes output projection.
    att=(1-a.alpha)*att+a.alpha*query
    x=x+self.linear(a.output_proj,att);y=layer.ln2(x)
    if self.spec.get('pack'):gate,value=(y@wffn.T).chunk(2,-1)
    else:gate,value=self.linear(f.w1,y),self.linear(f.w3,y)
    x=x+self.linear(f.w2,silu(gate)*value)
   h=self.base.ln_final(x if self.spec.get('all_logits')else x[:,-1]);return self.linear(self.base.lm_head,h).float()

 def prefill_attention_with_state(self,q,k,v,mask,state,a,ks=None,vs=None):
   backend='original';groups=a.num_heads//a.num_kv_heads
   if (backend=='triton_direct'and q.shape[-2]==1)or(self.table is not None and q.shape[-2]==1)or(ks is not None and q.shape[-2]==1):
    from hopfield_inference_triton_v1 import hopfield_decode
    return hopfield_decode(q,k,v,mask,state,a.alpha_prime,self.table,self.page_size,ks,vs)
   if backend=='native_gqa':
    b,h,n,d=q.shape;nk=k.shape[-2]
    scores=(q.reshape(b,a.num_kv_heads,groups,n,d)@k[:,:,None].transpose(-1,-2)).reshape(b,h,n,nk).float()/math.sqrt(d)
   else:
    k=k.repeat_interleave(groups,1);v=v.repeat_interleave(groups,1)
    scores=(q@k.transpose(-1,-2)).float()/math.sqrt(q.shape[-1])
   scores=scores.masked_fill(~mask,0)
   state=scores if state is None else a.alpha_prime*state+(1-a.alpha_prime)*scores
   if self.attention in ['sdpa','flash']:
    # SDPA consumes the already computed EMA as an additive bias. Zero Q/K
    # prevents a second ordinary QK term; explicit state is still returned.
    if backend=='native_gqa':v=v.repeat_interleave(groups,1)
    q0=torch.zeros_like(q);k0=torch.zeros((q.shape[0],q.shape[1],state.shape[-1],q.shape[-1]),device=q.device,dtype=q.dtype)
    # EMA remains FP32 for the next layer. Backend bias conversion is
    # an explicit mixed-precision approximation, checked by full tokens.
    bias=state.masked_fill(~mask,-torch.inf).to(q.dtype)
    if self.attention=='flash':
     from torch.nn.attention import sdpa_kernel,SDPBackend
     with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):out=F.scaled_dot_product_attention(q0,k0,v,attn_mask=bias,dropout_p=0,is_causal=False,scale=1.)
    else:out=F.scaled_dot_product_attention(q0,k0,v,attn_mask=bias,dropout_p=0,is_causal=False,scale=1.)
   else:
    probabilities=softmax(state.masked_fill(~mask,-torch.inf),-1).to(v.dtype)
    if backend=='native_gqa':
     b,h,n,nk=probabilities.shape
     out=(probabilities.reshape(b,a.num_kv_heads,groups,n,nk)@v[:,:,None]).reshape(b,h,n,a.head_dim)
    else:out=probabilities@v
   return out,state

def make_engine(model,spec,cases):
 if model.config.attention=='hopfield_official_v8'and spec.get('advanced'):
  engine=HopfieldEngine(model,spec,cases);engine.cases=cases;return engine
 return frozen_factory(model,spec,cases)
