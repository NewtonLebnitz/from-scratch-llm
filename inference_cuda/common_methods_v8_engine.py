"""Preserve FP32 RBF generator computation while using half precision backbone GEMMs."""
from common_methods_v7_engine import Engine,configure_model as configure_previous
from mlsystem.genlora import AdapterLinear

def configure_model(model,spec):
 result=configure_previous(model,spec)
 for module in model.modules():
  if isinstance(module,AdapterLinear)and module.kind=='rbf':
   module.gen_a.float();module.gen_b.float()
 result['generator_computation']='Original Gaussian generator FP32; generated fixed matrices cast once to backbone GEMM dtype'
 return result
