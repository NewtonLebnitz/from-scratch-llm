"""Exact greedy selection and token handoff optimization, isolated from production."""
import time,torch
from common_fusion_v1_engine import FusionEngine
from common_targeted_v3_engine import bucket_for,bucket_sizes

class DeliveryEngine(FusionEngine):
 def __init__(self,model,spec,cases,banned=None):
  super().__init__(model,spec,cases)
  ids=spec['banned_token_ids'];self.banned_mask=torch.zeros(self.base.config.vocab_size,device='cuda',dtype=torch.bool);self.banned_mask[ids]=True
  self.select_call=None;self.prefill_graphs={}
  self.host_pair=torch.empty(2,dtype=torch.long,pin_memory=True) if spec.get('pinned_delivery') else None
  self.delivery_event=torch.cuda.Event() if self.host_pair is not None else None
 def attend(self,q,k,v,mask):
  backend=self.spec.get('decode_attention','original')
  if backend=='native_gqa':
   n=self.decode_bucket
   return torch.nn.functional.scaled_dot_product_attention(q,k[:,:,:n],v[:,:,:n],attn_mask=mask[None,None,...,:n],dropout_p=0,is_causal=False,enable_gqa=True)
  if backend=='triton_direct' and q.shape[-2]==1 and k.shape[-2]==self.context:
   from common_decode_attention_v1 import decode_attention
   n=self.decode_bucket
   return decode_attention(q,k[:,:,:n],v[:,:,:n],mask[...,:n],self.spec.get('attention_warps',4))
  return super().attend(q,k,v,mask)
 def choose_tensor(self,z):
  finite=torch.isfinite(z).all()
  selected=z.masked_fill(self.banned_mask,-torch.inf).argmax()
  return torch.stack((selected,finite.to(torch.long)))
 def step_tensor(self,ids,pos):
  z=self.forward(ids,pos,fixed=True)
  return self.choose_tensor(z[0]) if self.spec.get('integrated_selection') else z
 def prefill_tensor(self,ids,positions):
  z=self.forward(ids,positions,fixed=False)
  return self.choose_tensor(z[0]) if self.spec.get('integrated_selection') else z
 @torch.no_grad()
 def prepare_acceleration(self):
  setup=self.prepare_core_acceleration()
  if self.spec.get('joint_selection') and not self.spec.get('integrated_selection'):
   t=time.perf_counter();self.select_call=torch.compile(self.choose_tensor,fullgraph=True,dynamic=False,mode='max-autotune')
   z=torch.zeros(self.base.config.vocab_size,device='cuda',dtype=torch.float32)
   for _ in range(3):self.select_call(z)
   torch.cuda.synchronize();setup['joint_selection_setup_seconds']=time.perf_counter()-t;setup['seconds']+=setup['joint_selection_setup_seconds']
  if self.spec.get('prefill_graph'):
   started=time.perf_counter();self.decode_bucket=self.context;self.position=0;captures=[]
   for n in self.prefill_lengths:
    ids=torch.ones((1,n),device='cuda',dtype=torch.long);pos=torch.arange(n,device='cuda');stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):self.prefill_call(ids,pos)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=self.prefill_call(ids,pos)
    graph.replay();torch.cuda.synchronize();self.prefill_graphs[n]=(graph,ids,pos,output);captures.append(n)
   setup['prefill_graph_seconds']=time.perf_counter()-started;setup['seconds']+=setup['prefill_graph_seconds'];setup['actual_prefill_graph_lengths']=captures
  return setup
 def last_logits(self,ids):
  n=ids.shape[1]
  if self.position==0 and n in self.prefill_graphs:
   self.decode_bucket=self.context;graph,static_ids,positions,result=self.prefill_graphs[n];static_ids.copy_(ids);graph.replay();self.position=n;return result
  return super().last_logits(ids)
 def generate(self,tok,case,settings,banned,stops):
  torch.cuda.synchronize();start=time.perf_counter();first=None;generated=[];steps=0;reason='length_limit';question_seconds=0
  q=None
  if self.question is not None:
   t=time.perf_counter();q=self.model.adapter.encode_question(tok,case['question']);torch.cuda.synchronize();question_seconds=time.perf_counter()-t
  self.reset(q);ids=case['prompt_ids'];x=torch.tensor([ids],device='cuda',dtype=torch.long)
  for _ in range(settings['max_new_tokens']):
   value=self.last_logits(x)
   if self.spec.get('joint_selection'):
    pair=value if self.spec.get('integrated_selection') else self.select_call(value[0].float())
    if self.host_pair is not None:
     if steps==0:self.host_pair.copy_(pair,non_blocking=True)
     self.delivery_event.record();self.delivery_event.synchronize();token,finite=self.host_pair.tolist()
    else:token,finite=pair.cpu().tolist()
    if not finite:raise RuntimeError('Nonfinite logits')
   else:
    z=value[0].float()
    if not torch.isfinite(z).all():raise RuntimeError('Nonfinite logits')
    z[banned]=-torch.inf;device_token=z.argmax();token=int(device_token)
   steps+=1
   if token in stops:reason='model_terminator';break
   generated.append(token)
   if first is None:first=time.perf_counter()-start
   x=pair[:1].reshape(1,1) if self.spec.get('joint_selection') else device_token.reshape(1,1)
  torch.cuda.synchronize();seconds=time.perf_counter()-start;decode_seconds=seconds-first if first is not None else None
  return dict(token_ids=generated,text=tok.decode(generated),stop_reason=reason,termination_token=token if reason=='model_terminator' else None,
   prompt_tokens=len(ids),tokens=len(generated),model_steps=steps,seconds=seconds,first_token_seconds=first,
   decode_tokens_per_second=(len(generated)-1)/decode_seconds if decode_seconds and len(generated)>1 else None,
   total_tokens_per_second=len(generated)/seconds,question_setup_seconds=question_seconds)

 @torch.no_grad()
 def prepare_core_acceleration(self):
  limits={}
  for key,value in [('recompile_limit',256),('accumulated_recompile_limit',1024)]:
   setattr(torch._dynamo.config,key,value);limits[key]=getattr(torch._dynamo.config,key)
  start=time.perf_counter()
  self.static_ids=torch.ones((1,1),device='cuda',dtype=torch.long)
  self.static_pos=torch.ones(1,device='cuda',dtype=torch.long)
  mode=self.spec.get('compile_mode','default')
  self.step_call=torch.compile(self.step_tensor,backend='inductor',fullgraph=True,dynamic=False,mode=mode)
  buckets=bucket_sizes(self.context) if self.spec.get('target_buckets') else [self.context]
  setup=[]
  for bucket in buckets:
   self.decode_bucket=bucket;t=time.perf_counter()
   self.step_call(self.static_ids,self.static_pos);torch.cuda.synchronize()
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):self.step_call(self.static_ids,self.static_pos)
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
    output=self.step_call(self.static_ids,self.static_pos)
    if self.host_pair is not None:self.host_pair.copy_(output,non_blocking=True)
   graph.replay();torch.cuda.synchronize()
   self.decode_graphs[bucket]=(graph,output)
   setup.append(dict(bucket=bucket,seconds=time.perf_counter()-t))
  boundary_checks=[]
  if self.spec.get('target_buckets'):
   for position in sorted({p for b in buckets for p in [b-1,b] if 0<=p<self.context}):
    # Causally valid identical pre-state, rather than sparse stale synthetic KV.
    self.position=0;self.decode_bucket=self.context
    for kc,vc in self.kv:kc.zero_();vc.zero_()
    self.forward(torch.ones((1,position),device='cuda',dtype=torch.long),torch.arange(position,device='cuda'))
    state=[(kc.clone(),vc.clone()) for kc,vc in self.kv]
    self.static_pos.fill_(position)
    graph,z=self.decode_graphs[self.context];graph.replay();reference=z.float().clone()
    for (kc,vc),(ks,vs) in zip(self.kv,state):kc.copy_(ks);vc.copy_(vs)
    bucket=bucket_for(position,self.context);graph,z=self.decode_graphs[bucket];graph.replay();candidate=z.float().clone()
    torch.cuda.synchronize()
    if self.spec.get('integrated_selection'):
     exact=bool(torch.equal(candidate,reference));finite=bool(candidate[1]==1 and reference[1]==1)
     assert exact and finite,'Selection boundary mismatch or nonfinite logits'
     boundary_checks.append(dict(position=position,bucket=bucket,exact_selected_token_and_finite_flag=exact,finite=finite,identical_causal_KV_snapshot=True))
    else:
     error=float((candidate-reference).abs().max());tolerance=1e-2 if self.dtype!=torch.float32 else 1e-4
     finite=bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all());assert finite
     boundary_checks.append(dict(position=position,bucket=bucket,max_logit_abs_error=error,argmax_equal=bool(candidate.argmax()==reference.argmax()),finite=finite,identical_causal_KV_snapshot=True))
    del state
  self.decode_bucket=self.context;self.position=0
  prefill_seconds=0
  if self.spec.get('target_prefill'):
   t=time.perf_counter()
   self.prefill_call=torch.compile(self.prefill_tensor,backend='inductor',fullgraph=True,dynamic=True,mode=mode)
   # Synthetic and all input-only prompt lengths, without consulting outputs.
   # Enumerated warmup is charged to setup, not hidden in request timing.
   for n in sorted(set([32,80]+self.prefill_lengths)):
    self.prefill_call(torch.ones((1,n),device='cuda',dtype=torch.long),torch.arange(n,device='cuda'))
   torch.cuda.synchronize();prefill_seconds=time.perf_counter()-t
  return dict(seconds=time.perf_counter()-start,mode=mode,bounded_compile_limits=limits,
   actual_model_context=self.context,decode_buckets=setup,prefill_compile_seconds=prefill_seconds,bucket_boundary_GPU_checks=boundary_checks,
   source='Measured profile motivated compiled prefill and valid-key decode buckets. Existing mechanisms adapted to this project; no weights or architecture changes.',
   prefill_warm_lengths=self.prefill_lengths if self.prefill_call is not None else [],
   shape_caveat='All workload prompt lengths plus synthetic32/80 warmed at setup. Raw rounds and compile warnings remain available; no pristine-cache cold claim.')
