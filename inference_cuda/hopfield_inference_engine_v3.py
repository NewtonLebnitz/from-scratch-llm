"""Correct eager-prefill selection boundary; frozen v2 model math is inherited."""
import torch
from hopfield_inference_engine_v2 import HopfieldEngine as FrozenHopfield,make_engine as frozen_factory,load_model

class HopfieldEngine(FrozenHopfield):
 def last_logits(self,ids):
  n=ids.shape[1]
  if self.spec.get('advanced')and(self.position==0 or n!=1):
   self.decode_seeded=False
   compiled=self.position==0 and(self.prefill_call is not None or n in self.prefill_graphs)
   if not compiled:
    if self.position+n>self.context:raise ValueError('Context overflow')
    self.decode_bucket=self.context
    positions=torch.arange(self.position,self.position+n,device=ids.device)
    z=self.forward(ids,positions);self.position+=n
    # Decode may stay integrated although prefill compilation is removed.
    return self.choose_tensor(z[0])if self.spec.get('integrated_selection')else z
  return super().last_logits(ids)

def make_engine(model,spec,cases):
 if model.config.attention=='hopfield_official_v8'and spec.get('advanced'):
  engine=HopfieldEngine(model,spec,cases);engine.cases=cases;return engine
 return frozen_factory(model,spec,cases)
