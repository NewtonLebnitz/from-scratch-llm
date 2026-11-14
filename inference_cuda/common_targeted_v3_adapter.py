"""Only new isolated process selects targeted engines; no production mutation."""
from common_confirm_v1_adapter import Engine as previous_engine,configure_model as previous_configure
from common_aux_v1_adapter import configure_model as aux_configure
from common_targeted_v3_engine import TargetedEngine

def configure_model(model,spec,root,tok):
 if spec['engine_suite']=='targeted-v3':return aux_configure(model,spec,root,tok)
 if spec['engine_suite']=='common-w8a8-v2':
  from common_w8a8_v2_adapter import configure_model as w8configure
  return w8configure(model,spec,root,tok)
 return previous_configure(model,spec,root,tok)

def Engine(model,spec,cases):
 if spec['engine_suite']=='targeted-v3':return TargetedEngine(model,spec,cases)
 if spec['engine_suite']=='common-w8a8-v2':
  from common_w8a8_v2_adapter import Engine as w8engine
  return w8engine(model,spec,cases)
 return previous_engine(model,spec,cases)
