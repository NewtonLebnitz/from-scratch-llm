"""Use the actual measured engine implementations with one common request harness."""
from common_methods_v8_engine import Engine as MainEngine,configure_model as main_configure
from common_aux_v1_adapter import Engine as AuxEngine,configure_model as aux_configure

def configure_model(model,spec,root,tok):
 if spec['engine_suite']=='common-aux-v1':return aux_configure(model,spec,root,tok)
 return main_configure(model,spec)

def Engine(model,spec,cases):
 if spec['engine_suite']=='common-aux-v1':return AuxEngine(model,spec,cases)
 return MainEngine(model,spec)
