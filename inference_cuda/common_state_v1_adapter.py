from common_delivery_v1_adapter import Engine as old_engine,configure_model as old_configure
from common_aux_v1_adapter import configure_model as aux_configure
from common_state_v1_engine import StateEngine

def configure_model(model,spec,root,tok):
 return aux_configure(model,spec,root,tok) if spec['engine_suite']=='state-v1' else old_configure(model,spec,root,tok)
def Engine(model,spec,cases):
 return StateEngine(model,spec,cases) if spec['engine_suite']=='state-v1' else old_engine(model,spec,cases)
