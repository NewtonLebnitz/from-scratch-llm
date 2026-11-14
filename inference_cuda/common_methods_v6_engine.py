"""Independent controlled adapters for existing inference methods; no training edits."""
import math,time
import torch
from torch.nn import functional as F
from mlsystem.genlora import AdapterLinear
from mlsystem.ols_rbf import RBFAdaptedModel
from mlsystem.components import Linear,softmax,silu

class Engine:
 def __init__(self,model,spec):
  self.model=model;self.base=model.base if isinstance(model,RBFAdaptedModel)else model
  assert self.base.config.attention=='standard','Do not substitute plain attention for Hopfield'
  self.spec=spec;self.context=self.base.context_length;self.cache=spec.get('cache','static');self.attention=spec.get('attention','sdpa')
  self.matrices={}
  for m in self.base.modules():
   if isinstance(m,AdapterLinear):self.matrices[id(m)]=tuple(x.detach()for x in m.matrices())
  self.question=None
  if isinstance(model,RBFAdaptedModel):
   self.question=torch.zeros(model.adapter.question_width,device='cuda',dtype=torch.float32)
   model.adapter._question_mean=self.question
  first=next(self.base.parameters());self.dtype=first.dtype
  self.kv=[]
  for layer in self.base.layers:
   a=layer.attn;shape=(1,a.num_kv_heads,self.context,a.head_dim)
   self.kv.append((torch.zeros(shape,device='cuda',dtype=self.dtype),torch.zeros(shape,device='cuda',dtype=self.dtype)))
  self.arange=torch.arange(self.context,device='cuda');self.position=0;self.prefix=[]
  self.step_call=None;self.graph=None;self.production=None
  if spec.get('production'):
   from mlsystem.runtime_decoding import make_decoder
   self.production_factory=make_decoder

 def linear(self,m,x):
  if isinstance(m,AdapterLinear):
   a,b=self.matrices[id(m)];return m.base_layer(x)+((x@a.T)@b.T)*m.scale
  return m(x)

 def attend(self,q,k,v,mask):
  groups=q.shape[1]//k.shape[1];k=k.repeat_interleave(groups,1);v=v.repeat_interleave(groups,1)
  if self.attention=='manual':
   scores=(q@k.transpose(-1,-2)).float()/math.sqrt(q.shape[-1]);p=softmax(scores.masked_fill(~mask,float('-inf')),-1)
   return p.to(v.dtype)@v
  if self.attention=='flash':
   from torch.nn.attention import sdpa_kernel,SDPBackend
   with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):return F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0,is_causal=q.shape[-2]>1)
  return F.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None],dropout_p=0,is_causal=False)

 def forward(self,ids,positions,*,fixed=False):
  b,n=ids.shape;x=self.base.token_embeddings(ids)
  if fixed:keys=self.arange;mask=keys[None,:]<=positions[:,None]
  elif self.cache=='none':keys=positions;mask=keys[None,:]<=positions[:,None]
  else:keys=torch.arange(self.position+n,device=ids.device);mask=keys[None,:]<=positions[:,None]
  for i,layer in enumerate(self.base.layers):
   a=layer.attn;y=layer.ln1(x)
   q=self.linear(a.q_proj,y).reshape(b,n,a.num_heads,a.head_dim).transpose(1,2)
   k=self.linear(a.k_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   v=self.linear(a.v_proj,y).reshape(b,n,a.num_kv_heads,a.head_dim).transpose(1,2)
   q,k=a.rope(q,positions),a.rope(k,positions)
   if self.cache!='none':
    if self.cache=='dynamic':
     if self.position:k=torch.cat((self.kv[i][0],k),2);v=torch.cat((self.kv[i][1],v),2)
     self.kv[i]=(k,v)
    else:
     kc,vc=self.kv[i];kc.index_copy_(2,positions,k);vc.index_copy_(2,positions,v)
     if fixed:k,v=kc,vc
     else:k,v=kc[:,:,:self.position+n],vc[:,:,:self.position+n]
   attended=self.attend(q,k,v,mask).transpose(1,2).reshape(b,n,-1)
   x=x+self.linear(a.output_proj,attended);y=layer.ln2(x);f=layer.ffn
   x=x+self.linear(f.w2,silu(self.linear(f.w1,y))*self.linear(f.w3,y))
  h=self.base.ln_final(x[:,-1]);z=self.linear(self.base.lm_head,h).float()
  if isinstance(self.model,RBFAdaptedModel):z=z+self.model.adapter(h)
  return z

 def step_tensor(self,ids,pos):return self.forward(ids,pos,fixed=True)

 def prepare_acceleration(self):
  if not(self.spec.get('compile')or self.spec.get('graph')):return dict(seconds=0)
  started=time.perf_counter();self.static_ids=torch.ones((1,1),device='cuda',dtype=torch.long);self.static_pos=torch.ones(1,device='cuda',dtype=torch.long)
  if self.spec.get('compile'):
   self.step_call=torch.compile(self.step_tensor,backend='inductor',fullgraph=True,dynamic=False,mode='default')
   self.step_call(self.static_ids,self.static_pos);torch.cuda.synchronize()
  else:
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):self.step_tensor(self.static_ids,self.static_pos)
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph):self.graph_output=self.step_tensor(self.static_ids,self.static_pos)
   self.graph.replay();torch.cuda.synchronize()
  return dict(seconds=time.perf_counter()-started,includes='First compile/kernel loading or graph warmup and capture, not steady-state timing')

 def reset(self,question_vector=None):
  self.position=0;self.prefix=[]
  if self.cache=='dynamic':self.kv=[None]*len(self.base.layers)
  if question_vector is not None:self.question.copy_(question_vector)
  if self.spec.get('production'):self.production=self.production_factory(self.model)

 def last_logits(self,ids):
  if self.spec.get('production'):return self.production.last_logits(ids)
  if self.cache=='none':
   self.prefix.extend(ids[0].tolist());ids=torch.tensor([self.prefix],device='cuda');positions=torch.arange(len(self.prefix),device='cuda')
   return self.forward(ids,positions)
  n=ids.shape[1]
  if self.position+n>self.context:raise ValueError('Context overflow')
  if self.position>0 and n==1 and self.graph is not None:
   self.static_ids.copy_(ids);self.static_pos.fill_(self.position);self.graph.replay();out=self.graph_output
  elif self.position>0 and n==1 and self.step_call is not None:
   self.static_pos.fill_(self.position);out=self.step_call(ids,self.static_pos)
  else:
   positions=torch.arange(self.position,self.position+n,device='cuda');out=self.forward(ids,positions)
  self.position+=n;return out

@torch.no_grad()
def configure_model(model,spec):
 base=model.base if isinstance(model,RBFAdaptedModel)else model
 if spec.get('merge')or spec.get('quant'):
  from mlsystem.genlora import merge
  merge(base)
 dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[spec['precision']]
 base.to(dtype=dtype)
 # OLS geometry and readout always retain their original FP32 computation.
 if isinstance(model,RBFAdaptedModel):model.adapter.float()
 quant=spec.get('quant')
 count=0
 if quant:
  import bitsandbytes as bnb
  targets=[(name,m)for name,m in base.named_modules()if isinstance(m,Linear)]
  for name,m in targets:
   out_dim,in_dim=m.weight.shape
   if quant=='int8':
    q=bnb.nn.Linear8bitLt(in_dim,out_dim,bias=False,has_fp16_weights=False,threshold=6.)
    q.weight=bnb.nn.Int8Params(m.weight.detach().cpu().half(),requires_grad=False,has_fp16_weights=False)
   elif quant=='nf4':
    q=bnb.nn.Linear4bit(in_dim,out_dim,bias=False,compute_dtype=torch.float16,compress_statistics=True,quant_type='nf4')
    q.weight=bnb.nn.Params4bit(m.weight.detach().cpu().half(),requires_grad=False,compress_statistics=True,quant_type='nf4')
   else:raise ValueError(quant)
   parent,_,leaf=name.rpartition('.');setattr(base.get_submodule(parent)if parent else base,leaf,q.to('cuda'));count+=1
 model.requires_grad_(False);return dict(quantized_linear_layers=count,precision=spec['precision'],adapter_geometry='FP32 unchanged')
