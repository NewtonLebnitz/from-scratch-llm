"""Correct mixed-precision derived GenLoRA matrices; original smoke source retained."""
from common_methods_v6_engine import Engine as BaseEngine,configure_model
from mlsystem.genlora import AdapterLinear

class Engine(BaseEngine):
 def __init__(self,model,spec):
  super().__init__(model,spec)
  self.matrices={k:tuple(m.to(self.dtype)for m in pair)for k,pair in self.matrices.items()}
 def linear(self,module,x):
  if isinstance(module,AdapterLinear):
   a,b=self.matrices[id(module)]
   return module.base_layer(x)+((x@a.T)@b.T)*module.scale
  return module(x)
