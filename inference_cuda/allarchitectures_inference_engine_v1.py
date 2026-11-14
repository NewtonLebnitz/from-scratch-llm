"""Generalized isolated ablation switches; original model forward is inherited."""
import torch
from common_state_v1_engine import StateEngine
from common_delivery_v1_engine import DeliveryEngine
from common_targeted_v3_engine import bucket_for
from hopfield_inference_engine_v4 import make_engine as hop_factory,load_model
class StandardAblationEngine(StateEngine):
 def __init__(self,model,spec,cases):
  DeliveryEngine.__init__(self,model,spec,cases);self.decode_seeded=False
 def step_tensor(self,ids,pos):
  z=self.forward(ids,pos,fixed=True)
  if not self.spec.get('integrated_selection'):return z
  pair=self.choose_tensor(z[0])
  if self.spec.get('gpu_feedback'):ids.copy_(pair[:1].reshape(1,1))
  if self.spec.get('gpu_position'):pos.add_(1)
  return pair
 def last_logits(self,ids):
  n=ids.shape[1]
  if self.position+n>self.context:raise ValueError('Context overflow')
  if self.position==0 or n!=1:
   self.decode_seeded=False
   if self.position==0 and(self.prefill_call is not None or n in self.prefill_graphs):return DeliveryEngine.last_logits(self,ids)
   self.decode_bucket=self.context
   positions=torch.arange(self.position,self.position+n,device=ids.device)
   z=self.forward(ids,positions);self.position+=n
   return self.choose_tensor(z[0])if self.spec.get('integrated_selection')else z
  bucket=bucket_for(self.position,self.context)if self.spec.get('target_buckets')else self.context
  graph,result=self.decode_graphs[bucket]
  if not self.decode_seeded or not self.spec.get('gpu_feedback'):self.static_ids.copy_(ids)
  if not self.decode_seeded or not self.spec.get('gpu_position'):self.static_pos.fill_(self.position)
  graph.replay();self.position+=n;self.decode_seeded=True;return result

def make_engine(model,spec,cases):
 if model.config.attention=='standard' and spec.get('engine_suite')=='state-v1':
  return StandardAblationEngine(model,spec,cases)
 return hop_factory(model,spec,cases)
