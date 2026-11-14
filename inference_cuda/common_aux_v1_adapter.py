"""Explicit selection of additional common-method mechanisms; derived models only."""
import torch
from mlsystem.genlora import merge,AdapterLinear
from mlsystem.ols_rbf import RBFAdaptedModel
from common_methods_v8_engine import Engine as StandardEngine,configure_model as standard_configure
from common_cache_principles_v1 import PrefixEngine,PagedEngine
from common_kv_quant_v1 import KV8Engine
from common_compiled_graph_v1 import CompiledGraphEngine
from common_speculation_v1 import SpeculativeEngine
from common_calibration_v1 import calibrate_and_convert

def configure_model(model,spec,root,tok):
 base=model.base if isinstance(model,RBFAdaptedModel)else model
 if spec.get('cal_quant')or spec.get('merged_reference'):merge(base)
 # Preserve exact master generator tensors before any recursive backbone .to(dtype).
 # Casting the whole backbone to half and then generator.float() restores dtype,
 # but does NOT restore original values; the earlier primary variant is disclosed.
 saved=[]
 if spec.get('preserve_generator_values'):
  for module in base.modules():
   if isinstance(module,AdapterLinear)and module.kind=='rbf':
    for generator in [module.gen_a,module.gen_b]:saved.append((generator,{k:v.detach().clone()for k,v in generator.state_dict().items()}))
 result=standard_configure(model,spec)
 for generator,state in saved:generator.float();generator.load_state_dict(state)
 if saved:result['generator_master_values']='Exact pre-cast FP32 state restored and CPU unit-checked; final A/B cast only at Engine boundary'
 if spec.get('cal_quant'):result['calibration']=calibrate_and_convert(model,spec['cal_quant'],root,tok)
 return result

class TF32Engine(StandardEngine):
 pass  # Per-request backend selection occurs outside the timed generation loop.

class ProductionAlgorithmPort:
 def __init__(self,model):
  self.model=model;self.base=model.base if isinstance(model,RBFAdaptedModel)else model
  self.question=None
  if isinstance(model,RBFAdaptedModel):
   self.question=torch.zeros(model.adapter.question_width,device='cuda');model.adapter._question_mean=self.question
  if any(isinstance(m,AdapterLinear)for m in self.base.modules()):
   from mlsystem.runtime_reference_v3 import MergedPlanDecoder
  else:
   from mlsystem.runtime_decoding import MergedPlanDecoder
  self.factory=MergedPlanDecoder
 def reset(self,question_vector=None):
  if question_vector is not None:self.question.copy_(question_vector)
  self.decoder=self.factory(self.model)
 def prepare_acceleration(self):return dict(seconds=0)
 def last_logits(self,ids):return self.decoder.last_logits(ids)

def Engine(model,spec,cases):
 mode=spec.get('aux_mode')
 if spec.get('allow_tf32'):return TF32Engine(model,spec)
 if spec.get('compiled_graph'):return CompiledGraphEngine(model,spec)
 if spec.get('speculation'):return SpeculativeEngine(model,spec)
 if mode=='prefix':return PrefixEngine(model,spec,[case['system_prefix_ids']for case in cases])
 if mode=='paged':return PagedEngine(model,spec)
 if mode=='kv_int8':return KV8Engine(model,spec)
 if mode=='production_algorithm_port':return ProductionAlgorithmPort(model)
 return StandardEngine(model,spec)
