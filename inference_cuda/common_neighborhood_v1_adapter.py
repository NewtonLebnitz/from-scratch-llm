"""Optional prepared inference-mode facade; no training/default model changes."""
from common_state_v1_adapter import Engine as state_engine,configure_model as state_configure

def configure_model(model,spec,root,tok):
 if spec.get('inference_mode'):
  import torch
  with torch.inference_mode():return state_configure(model,spec,root,tok)
 return state_configure(model,spec,root,tok)

class InferenceModeEngine:
 def __init__(self,inner):self.inner=inner
 def __getattr__(self,name):return getattr(self.inner,name)
 def prepare_acceleration(self):
  import torch
  with torch.inference_mode():out=self.inner.prepare_acceleration()
  return dict(out,inference_mode=True)
 def generate(self,*args,**kwargs):
  import torch
  with torch.inference_mode():return self.inner.generate(*args,**kwargs)

def Engine(model,spec,cases):
 if spec.get('inference_mode'):
  import torch
  with torch.inference_mode():inner=state_engine(model,spec,cases)
  assert hasattr(inner,'generate'),'Inference-mode variant requires exact streaming engine'
  return InferenceModeEngine(inner)
 return state_engine(model,spec,cases)
