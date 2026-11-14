"""Shared-prefix KV and page-table KV mechanisms for the handwritten model."""
import math,time,torch
from common_methods_v8_engine import Engine
from mlsystem.components import silu
from mlsystem.ols_rbf import RBFAdaptedModel

class PrefixEngine(Engine):
 def __init__(self,model,spec,prefixes):
  super().__init__(model,spec);assert self.cache=='static'and prefixes
  # Each entry is a fixed system prelude; never cache a question or generated answer.
  if isinstance(prefixes[0],int):prefixes=[prefixes]
  self.saved_prefixes=[]
  for prefix in sorted({tuple(p)for p in prefixes},key=len,reverse=True):
   assert prefix;Engine.reset(self)
   Engine.last_logits(self,torch.tensor([prefix],device='cuda',dtype=torch.long))
   kv=[(k[:,:,:len(prefix)].clone(),v[:,:,:len(prefix)].clone())for k,v in self.kv]
   self.saved_prefixes.append((prefix,kv))
  self.reset()
 def reset(self,question_vector=None):
  super().reset(question_vector);self.first_request_input=True
 def last_logits(self,ids):
  if self.first_request_input:
   tokens=ids[0].tolist();matched=False
   for prefix,stored in self.saved_prefixes:
    length=len(prefix)
    if tokens[:length]!=list(prefix):continue
    assert length<len(tokens);self.position=length
    for (k,v),(pk,pv)in zip(self.kv,stored):
     k[:,:,:length].copy_(pk);v[:,:,:length].copy_(pv)
    ids=ids[:,length:];matched=True;break
   assert matched,'No system-only prefix entry for this prompt'
   self.first_request_input=False
  return super().last_logits(ids)

class PagedEngine(Engine):
 def __init__(self,model,spec,page_size=16):
  super().__init__(model,spec);assert self.cache=='static'and self.context%page_size==0
  self.page_size=page_size
  # Nonidentity physical mapping also checks correctness across page boundaries.
  self.table=torch.arange(self.context//page_size-1,-1,-1,device='cuda',dtype=torch.long)
 def prepare_acceleration(self):
  from common_paged_triton_v1 import paged_decode
  begin=time.perf_counter();a=self.base.layers[0].attn
  q=torch.zeros((1,a.num_heads,1,a.head_dim),device=self.arange.device,dtype=self.dtype)
  # Every legal length, not test-selected prompts: also warms scalar alignment specializations.
  for length in range(1,self.context+1):paged_decode(q,*self.kv[0],self.table,length,self.page_size)
  torch.cuda.synchronize()
  return dict(seconds=time.perf_counter()-begin,warmed_valid_lengths=self.context,decode_tile='next power of two of actual valid length, minimum16',JIT_outside_formal_timer=True)
 def forward(self,ids,positions,*,fixed=False):
  assert not fixed
  b,n=ids.shape;assert b==1;x=self.base.token_embeddings(ids);end=self.position+n
  logical=torch.arange(end,device='cuda');mask=logical[None,:]<=positions[:,None]
  physical=self.table[positions//self.page_size]*self.page_size+positions%self.page_size
  key_physical=self.table[logical//self.page_size]*self.page_size+logical%self.page_size
  for i,layer in enumerate(self.base.layers):
   a=layer.attn;y=layer.ln1(x)
   q=self.linear(a.q_proj,y).reshape(b,n,a.num_heads,a.head_dim).transpose(1,2)
   k=self.linear(a.k_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   v=self.linear(a.v_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   q,k=a.rope(q,positions),a.rope(k,positions);kc,vc=self.kv[i]
   kc.index_copy_(2,physical,k);vc.index_copy_(2,physical,v)
   if n==1:
    from common_paged_triton_v1 import paged_decode
    attended=paged_decode(q,kc,vc,self.table,end,self.page_size)
   else:attended=self.attend(q,kc.index_select(2,key_physical),vc.index_select(2,key_physical),mask)
   attended=attended.transpose(1,2).reshape(b,n,-1)
   x=x+self.linear(a.output_proj,attended);y=layer.ln2(x);f=layer.ffn
   x=x+self.linear(f.w2,silu(self.linear(f.w1,y))*self.linear(f.w3,y))
  h=self.base.ln_final(x[:,-1]);z=self.linear(self.base.lm_head,h).float()
  if isinstance(self.model,RBFAdaptedModel):z=z+self.model.adapter(h)
  return z

class VerificationEngine(Engine):
 def forward(self,ids,positions,*,fixed=False):
  assert self.cache=='static'and not fixed
  b,n=ids.shape;x=self.base.token_embeddings(ids);end=self.position+n
  keys=torch.arange(end,device=ids.device);mask=keys[None,:]<=positions[:,None]
  for i,layer in enumerate(self.base.layers):
   a=layer.attn;y=layer.ln1(x)
   q=self.linear(a.q_proj,y).reshape(b,n,a.num_heads,a.head_dim).transpose(1,2)
   k=self.linear(a.k_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   v=self.linear(a.v_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   q,k=a.rope(q,positions),a.rope(k,positions);kc,vc=self.kv[i]
   kc.index_copy_(2,positions,k);vc.index_copy_(2,positions,v)
   attended=self.attend(q,kc[:,:,:end],vc[:,:,:end],mask).transpose(1,2).reshape(b,n,-1)
   x=x+self.linear(a.output_proj,attended);y=layer.ln2(x);f=layer.ffn
   x=x+self.linear(f.w2,silu(self.linear(f.w1,y))*self.linear(f.w3,y))
  h=self.base.ln_final(x);z=self.linear(self.base.lm_head,h).float()
  if isinstance(self.model,RBFAdaptedModel):z=z+self.model.adapter(h)
  return z
 def all_logits(self,ids):
  positions=torch.arange(self.position,self.position+ids.shape[1],device=ids.device)
  z=self.forward(ids,positions);self.position+=ids.shape[1];return z
 def last_logits(self,ids):return self.all_logits(ids)[:,-1]
