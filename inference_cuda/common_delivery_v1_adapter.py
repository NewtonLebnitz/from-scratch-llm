from common_fusion_v1_adapter import Engine as old_engine,configure_model as old_configure
from common_aux_v1_adapter import configure_model as aux_configure
from common_delivery_v1_engine import DeliveryEngine

def configure_model(model,spec,root,tok):
 return aux_configure(model,spec,root,tok) if spec['engine_suite']=='delivery-v1' else old_configure(model,spec,root,tok)
def Engine(model,spec,cases):
 return DeliveryEngine(model,spec,cases) if spec['engine_suite']=='delivery-v1' else old_engine(model,spec,cases)
