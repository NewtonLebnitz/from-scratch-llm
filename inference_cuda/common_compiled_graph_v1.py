"""Common Inductor operator fusion plus actual external CUDA Graph replay."""
import time,torch
from common_methods_v8_engine import Engine

def configure_compile_limits():
 # Multiple co-resident model/precision/backend variants must not silently hit
 # Dynamo's small default guard-cache limit. Never reset live compiled functions.
 config=torch._dynamo.config;result={}
 for current,legacy,value in [('recompile_limit','cache_size_limit',64),('accumulated_recompile_limit','accumulated_cache_size_limit',256)]:
  key=current if hasattr(config,current)else legacy if hasattr(config,legacy)else None
  if key is None:raise RuntimeError('Required bounded Dynamo cache control unavailable: '+current)
  setattr(config,key,value);result[key]=getattr(config,key)
 return result

class CompiledGraphEngine(Engine):
 @torch.no_grad()
 def prepare_acceleration(self):
  limits=configure_compile_limits();start=time.perf_counter();self.static_ids=torch.ones((1,1),device='cuda',dtype=torch.long);self.static_pos=torch.ones(1,device='cuda',dtype=torch.long)
  mode=self.spec.get('compile_mode','default')
  self.step_call=torch.compile(self.step_tensor,backend='inductor',fullgraph=True,dynamic=False,mode=mode)
  self.step_call(self.static_ids,self.static_pos);torch.cuda.synchronize();compile_seconds=time.perf_counter()-start
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):self.step_call(self.static_ids,self.static_pos)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();capture_start=time.perf_counter();self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph):self.graph_output=self.step_call(self.static_ids,self.static_pos)
  self.graph.replay();torch.cuda.synchronize()
  return dict(seconds=time.perf_counter()-start,compile_seconds=compile_seconds,capture_seconds=time.perf_counter()-capture_start,
   mode=mode,bounded_compile_limits=limits,actual_external_CUDA_graph=True,scope='Common existing compiler+graph combination, not a new proposed method; exact output still requires complete workload verification')
