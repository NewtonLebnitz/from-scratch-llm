"""Inference-only official Hopfield adaptation. No training or checkpoint mutation.
State is recreated across layers for each query, not persisted across tokens.
"""
import math,time,torch
from torch.nn import functional as F
from common_state_v1_engine import StateEngine
from common_methods_v6_engine import Engine as OrdinaryEngine
from common_compiled_graph_v1 import CompiledGraphEngine
from common_targeted_v3_engine import bucket_for
from mlsystem.components import silu,softmax

class HopfieldEngine(StateEngine):
 def __init__(self,model,spec,cases):
  self.model=self.base=model;assert model.config.attention=='hopfield_official_v8'
  self.spec=spec;self.context=model.context_length;self.cache=spec.get('cache','static');self.attention=spec.get('attention','manual')
  first=next(model.parameters());self.dtype=first.dtype;self.device=first.device;self.question=None;self.matrices={}
  self.kv=[];self.kv_scales=[];bs=spec.get('batch_size',1)
  for layer in model.layers:
   a=layer.attn;shape=(bs,a.num_kv_heads,self.context,a.head_dim)
   self.kv.append(tuple(torch.zeros(shape,device=self.device,dtype=torch.int8 if spec.get('aux_mode')=='kv_int8' else self.dtype)for _ in range(2)))
   if spec.get('aux_mode')=='kv_int8':self.kv_scales.append(tuple(torch.ones(shape[:-1],device=self.device,dtype=torch.float32)for _ in range(2)))
  self.arange=torch.arange(self.context,device=self.device);self.position=0;self.prefix=[]
  self.step_call=None;self.graph=None;self.production=None;self.decode_bucket=self.context;self.decode_graphs={};self.prefill_call=None;self.prefill_graphs={}
  self.prefill_lengths=sorted({len(row['prompt_ids'])for row in cases});self.decode_seeded=False
  self.banned_mask=torch.zeros(model.config.vocab_size,device=self.device,dtype=torch.bool);self.banned_mask[spec.get('banned_token_ids',[])]=True
  self.select_call=None;self.host_pair=torch.empty(2,dtype=torch.long,pin_memory=True)if spec.get('pinned_delivery')else None
  self.delivery_event=torch.cuda.Event()if self.host_pair is not None else None
  self.page_size=16;self.table=torch.arange(self.context//16-1,-1,-1,device=self.device,dtype=torch.long)if spec.get('aux_mode')=='paged'else None
  self.packed=[]
  for layer in model.layers:
   a=layer.attn;f=layer.ffn
   self.packed.append((torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight],0).detach(),torch.cat([a.k_proj.weight,a.v_proj.weight],0).detach(),torch.cat([f.w1.weight,f.w3.weight],0).detach())if spec.get('pack')else None)
  if spec.get('shared_rope'):
   first=model.layers[0].attn.rope
   for l in model.layers:assert torch.equal(first.cos,l.attn.rope.cos)and torch.equal(first.sin,l.attn.rope.sin)
  self.saved_prefixes=[];self.first_request_input=True
 def linear(self,m,x):return m(x)
 def reset(self,question_vector=None):
  self.position=0;self.prefix=[];self.decode_seeded=False;self.first_request_input=True
  if self.cache=='dynamic':self.kv=[None]*len(self.base.layers)
  if self.spec.get('production'):
   from mlsystem.runtime_decoding import MergedPlanDecoder
   self.production=MergedPlanDecoder(self.model)
 def attention_with_state(self,q,k,v,mask,state,a,ks=None,vs=None):
  backend=self.spec.get('decode_attention','original');groups=a.num_heads//a.num_kv_heads
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
   att,state=self.attention_with_state(q,k,v,mask[-1:]if last else mask,state,a,km,vm)
   att=att.transpose(1,2).reshape(b,nq,-1)
   # Official v8: visible-state mixture precedes output projection.
   att=(1-a.alpha)*att+a.alpha*query
   x=x+self.linear(a.output_proj,att);y=layer.ln2(x)
   if self.spec.get('pack'):gate,value=(y@wffn.T).chunk(2,-1)
   else:gate,value=self.linear(f.w1,y),self.linear(f.w3,y)
   x=x+self.linear(f.w2,silu(gate)*value)
  h=self.base.ln_final(x if self.spec.get('all_logits')else x[:,-1]);return self.linear(self.base.lm_head,h).float()
 def step_tensor(self,ids,pos):
  z=self.forward(ids,pos,fixed=True)
  if not self.spec.get('integrated_selection'):return z
  pair=self.choose_tensor(z[0])
  if self.spec.get('gpu_feedback'):ids.copy_(pair[:1].reshape(1,1))
  if self.spec.get('gpu_position'):pos.add_(1)
  return pair
 def last_logits(self,ids):
  if self.spec.get('production'):return self.production.last_logits(ids)
  if self.cache!='none' and self.position+ids.shape[1]>self.context:raise ValueError('Context overflow')
  if self.spec.get('aux_mode')=='prefix'and self.first_request_input:
   tokens=ids[0].tolist()
   for prefix,stored in self.saved_prefixes:
    if tokens[:len(prefix)]==list(prefix):
     self.position=len(prefix)
     for (k,v),(pk,pv)in zip(self.kv,stored):k[:,:,:len(prefix)].copy_(pk);v[:,:,:len(prefix)].copy_(pv)
     ids=ids[:,len(prefix):];break
   else:raise ValueError('No system-only prefix entry')
   self.first_request_input=False
  if self.spec.get('advanced'):
   if self.position==0 or ids.shape[1]!=1:return super(StateEngine,self).last_logits(ids)
   n=ids.shape[1];bucket=bucket_for(self.position,self.context)if self.spec.get('target_buckets')else self.context
   graph,result=self.decode_graphs[bucket]
   if not self.decode_seeded or not self.spec.get('gpu_feedback'):self.static_ids.copy_(ids)
   if not self.decode_seeded or not self.spec.get('gpu_position'):self.static_pos.fill_(self.position)
   graph.replay();self.position+=n;self.decode_seeded=True;return result
  return OrdinaryEngine.last_logits(self,ids)
 @torch.no_grad()
 def prepare_acceleration(self):
  if self.spec.get('aux_mode')=='prefix':
   for prefix in sorted({tuple(c['system_prefix_ids'])for c in self.cases},key=len,reverse=True):
    self.reset();OrdinaryEngine.last_logits(self,torch.tensor([prefix],device=self.device))
    self.saved_prefixes.append((prefix,[(k[:,:,:len(prefix)].clone(),v[:,:,:len(prefix)].clone())for k,v in self.kv]))
   self.reset()
  if self.spec.get('advanced'):return super().prepare_acceleration()
  if self.spec.get('compiled_graph'):return CompiledGraphEngine.prepare_acceleration(self)
  return OrdinaryEngine.prepare_acceleration(self)

class OrdinaryHopfieldEngine(HopfieldEngine):
 # Hide the inherited streaming generator: ordinary baselines must use the same
 # reference request loop, including constructing each next-token GPU input.
 generate=None

class HopfieldVerification(HopfieldEngine):
 def all_logits(self,ids):
  positions=torch.arange(self.position,self.position+ids.shape[1],device=ids.device)
  z=self.forward(ids,positions);self.position+=ids.shape[1];return z
 def last_logits(self,ids):return self.all_logits(ids)[:,-1]

class HopfieldSpeculation:
 def __new__(cls,model,spec,cases):
  import copy
  from torch import nn
  from common_speculation_v1 import SpeculativeEngine
  obj=object.__new__(SpeculativeEngine);obj.model=model;obj.spec=spec
  target_spec={**spec,'cache':'static','attention':'manual','all_logits':True,'last_query':False}
  obj.target=HopfieldVerification(model,target_spec,cases);obj.base=model;obj.question=None;obj.draft=None
  if spec['speculation']=='layer_skip':
   partial=copy.copy(model);partial._modules=model._modules.copy();partial.layers=nn.ModuleList(list(model.layers)[:len(model.layers)//2]);partial.eval()
   obj.draft=OrdinaryHopfieldEngine(partial,{**target_spec,'all_logits':False},cases)
  obj.draft_depth=len(obj.draft.base.layers)if obj.draft else 0
  return obj

def make_engine(model,spec,cases):
 if model.config.attention=='standard':
  from common_neighborhood_v1_adapter import Engine
  return Engine(model,spec,cases)
 if spec.get('speculation'):return HopfieldSpeculation(model,spec,cases)
 cls=HopfieldEngine if spec.get('advanced')else OrdinaryHopfieldEngine
 engine=cls(model,spec,cases);engine.cases=cases
 return engine


def load_model(root,path,device):
 """Read the isolated full SFT artifact without registering or moving it."""
 if path!='runtime/experiments/hopfield-official-v8/full-sft/best.pt':
  from mlsystem.sft import load_sft
  return load_sft(root,path,device)
 from pathlib import Path
 from mlsystem.sft import file_sha,restore_payload
 from mlsystem.config import ModelConfig
 from mlsystem.model import LanguageModel
 p=(Path(root)/path).resolve();assert p.is_relative_to(Path(root).resolve())
 assert file_sha(p)=='798353cb18836e330d0f256d79599742e8b1a68b81cfd3332b173fdeea4798ce'
 state=torch.load(p,map_location='cpu',weights_only=True)
 assert state['format']=='mlsystem-sft-v1' and state['settings']['route']=='full'
 parent=Path(root)/state['identity']['parent_path'];assert file_sha(parent)==state['identity']['parent_sha256']
 cfg=ModelConfig(**state['model_config']);assert cfg.attention=='hopfield_official_v8'
 model=LanguageModel(cfg);restore_payload(model,'full',state['payload']);return model.to(device).eval(),state
