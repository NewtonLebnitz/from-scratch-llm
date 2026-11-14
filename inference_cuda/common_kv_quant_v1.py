"""Inference-only per-token/head INT8 KV compression, original model tensors untouched."""
import time,torch
from common_methods_v8_engine import Engine
from mlsystem.components import silu
from mlsystem.ols_rbf import RBFAdaptedModel

class KV8Engine(Engine):
 def __init__(self,model,spec):
  super().__init__(model,spec);assert self.cache=='static';self.kv=[];self.scales=[]
  for layer in self.base.layers:
   a=layer.attn;shape=(1,a.num_kv_heads,self.context,a.head_dim);s=(1,a.num_kv_heads,self.context)
   self.kv.append((torch.empty(shape,device='cuda',dtype=torch.int8),torch.empty(shape,device='cuda',dtype=torch.int8)))
   self.scales.append((torch.empty(s,device='cuda'),torch.empty(s,device='cuda')))
 def prepare_acceleration(self):
  from common_kv_quant_triton_v1 import kv8_decode
  begin=time.perf_counter();a=self.base.layers[0].attn
  q=torch.zeros((1,a.num_heads,1,a.head_dim),device=self.arange.device,dtype=self.dtype)
  for item in self.kv[0]:item.zero_()
  for item in self.scales[0]:item.fill_(1)
  for length in range(1,self.context+1):kv8_decode(q,*self.kv[0],*self.scales[0],length)
  torch.cuda.synchronize();count=lambda values:sum(x.numel()*x.element_size()for pair in values for x in pair)
  return dict(seconds=time.perf_counter()-begin,kv_bytes=count(self.kv)+count(self.scales),compressed_kv_bytes=count(self.kv),scale_bytes=count(self.scales),scope='Actual compressed KV storage, not whole-model memory',warmed_valid_lengths=self.context,decode_tile='next power of two of actual valid length, minimum16',JIT_outside_formal_timer=True)
 def forward(self,ids,positions,*,fixed=False):
  assert not fixed;batch,n=ids.shape;assert batch==1 and(n==1 or self.position==0)
  x=self.base.token_embeddings(ids);end=self.position+n
  for index,layer in enumerate(self.base.layers):
   a=layer.attn;y=layer.ln1(x)
   q=self.linear(a.q_proj,y).reshape(batch,n,a.num_heads,a.head_dim).transpose(1,2)
   k=self.linear(a.k_proj,y).reshape(batch,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   v=self.linear(a.v_proj,y).reshape(batch,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   q,k=a.rope(q,positions),a.rope(k,positions);kc,vc=self.kv[index];ks,vs=self.scales[index]
   sk=k.abs().amax(-1).clamp_min(1e-12)/127;sv=v.abs().amax(-1).clamp_min(1e-12)/127
   kc.index_copy_(2,positions,(k/sk[...,None]).round().clamp(-127,127).to(torch.int8));vc.index_copy_(2,positions,(v/sv[...,None]).round().clamp(-127,127).to(torch.int8))
   ks.index_copy_(2,positions,sk);vs.index_copy_(2,positions,sv)
   if self.position:
    from common_kv_quant_triton_v1 import kv8_decode
    attended=kv8_decode(q,kc,vc,ks,vs,end)
   else:
    mask=positions[None,:]<=positions[:,None];attended=self.attend(q,k,v,mask)
   attended=attended.transpose(1,2).reshape(batch,n,-1);x=x+self.linear(a.output_proj,attended)
   y=layer.ln2(x);f=layer.ffn;x=x+self.linear(f.w2,silu(self.linear(f.w1,y))*self.linear(f.w3,y))
  h=self.base.ln_final(x[:,-1]);z=self.linear(self.base.lm_head,h).float()
  if isinstance(self.model,RBFAdaptedModel):z=z+self.model.adapter(h)
  return z
