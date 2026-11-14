"""GPU-maintained decode input/position, isolated exact-streaming experiment."""
import time,torch
from common_delivery_v1_engine import DeliveryEngine
from common_targeted_v3_engine import bucket_for,bucket_sizes

class StateEngine(DeliveryEngine):
 def __init__(self,model,spec,cases):
  assert spec.get('integrated_selection') and spec.get('target_buckets'),'State variants require integrated exact selection and decode graphs'
  super().__init__(model,spec,cases);self.decode_seeded=False
 def reset(self,question_vector=None):
  super().reset(question_vector);self.decode_seeded=False
 def step_tensor(self,ids,pos):
  # All model reads precede selection; writes are in the same compiled graph.
  pair=self.choose_tensor(self.forward(ids,pos,fixed=True)[0])
  if self.spec.get('gpu_feedback'):ids.copy_(pair[:1].reshape(1,1))
  if self.spec.get('gpu_position'):pos.add_(1)
  return pair
 def last_logits(self,ids):
  n=ids.shape[1]
  if self.position+n>self.context:raise ValueError('Context overflow')
  if self.position==0 or n!=1:
   self.decode_seeded=False;return super().last_logits(ids)
  bucket=bucket_for(self.position,self.context);graph,result=self.decode_graphs[bucket]
  if not self.decode_seeded or not self.spec.get('gpu_feedback'):self.static_ids.copy_(ids)
  if not self.decode_seeded or not self.spec.get('gpu_position'):self.static_pos.fill_(self.position)
  graph.replay();self.position+=1;self.decode_seeded=True;return result
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
    self.static_pos.fill_(position);self.static_ids.fill_(1)
    graph,z=self.decode_graphs[self.context];graph.replay();reference=z.float().clone()
    for (kc,vc),(ks,vs) in zip(self.kv,state):kc.copy_(ks);vc.copy_(vs)
    self.static_pos.fill_(position);self.static_ids.fill_(1)
    bucket=bucket_for(position,self.context);graph,z=self.decode_graphs[bucket];graph.replay();candidate=z.float().clone()
    torch.cuda.synchronize()
    if self.spec.get('integrated_selection'):
     exact=bool(torch.equal(candidate,reference));finite=bool(candidate[1]==1 and reference[1]==1)
     assert exact and finite,'Selection boundary mismatch or nonfinite logits'
     actual_position=int(self.static_pos[0]);actual_input=int(self.static_ids[0,0]);expected_position=position+int(bool(self.spec.get('gpu_position')));expected_input=int(candidate[0]) if self.spec.get('gpu_feedback') else 1
     assert actual_position==expected_position and actual_input==expected_input,'GPU feedback/position transition mismatch'
     boundary_checks.append(dict(position=position,bucket=bucket,exact_selected_token_and_finite_flag=exact,finite=finite,identical_causal_KV_snapshot=True,actual_next_position=actual_position,expected_next_position=expected_position,actual_input_token=actual_input,expected_input_token=expected_input))
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
