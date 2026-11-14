"""Correct L20/Torch _int_mm M>16 alignment in isolated follow-up only.

The frozen common-aux-v1 probe used M=16 and is retained as failed. These
modules do not replace any production or previously frozen source on disk.
"""
import torch
from common_quant_principles_v1 import W8A8Linear as OriginalW8A8Linear

class W8A8Linear(OriginalW8A8Linear):
 def forward(self,x):
  shape=x.shape;flat=x.reshape(-1,shape[-1]).float()/self.input_scale;rows=len(flat)
  # This observed Torch CUDA backend requires M>16. Multiples of32 cover both
  # that lower bound and the int8 Tensor Core row-alignment requirement.
  padded=max(32,((rows+31)//32)*32)
  if padded!=rows:flat=torch.nn.functional.pad(flat,(0,0,0,padded-rows))
  sx=flat.abs().amax(-1).clamp_min(1e-8)/127
  values=(flat/sx[:,None]).round().clamp(-127,127).to(torch.int8)
  result=torch._int_mm(values,self.weight8.T).float()*sx[:,None]*self.scales[None]
  return result[:rows].to(x.dtype).reshape(*shape[:-1],self.weight8.shape[0])

@torch.no_grad()
def check():
 torch.manual_seed(1817);results=[]
 for smooth in [False,True]:
  for rows in [1,7,16,17,19,32,65]:
   weight=torch.randn((128,512),device='cuda',dtype=torch.float16)*.2
   act_max=torch.rand(512,device='cuda')+1 if smooth else None
   module=W8A8Linear(weight,act_max);x=torch.randn((rows,512),device='cuda',dtype=torch.float16)
   result=module(x);flat=x.float()/module.input_scale;sx=flat.abs().amax(-1).clamp_min(1e-8)/127
   values=(flat/sx[:,None]).round().clamp(-127,127).to(torch.int8)
   integer_reference=values.cpu().int()@module.weight8.cpu().int().T
   reference=(integer_reference.float()*sx.cpu()[:,None]*module.scales.cpu()[None]).half().cuda()
   absolute=float((result-reference).abs().max())
   assert torch.isfinite(result).all()and torch.allclose(result,reference,atol=.002,rtol=.002),(rows,smooth,absolute)
   results.append(dict(rows=rows,padded_rows=max(32,((rows+31)//32)*32),smoothquant=smooth,max_abs=absolute))
 torch.cuda.synchronize()
 return dict(passed=True,W8A8=dict(available=True,checks=results),scope='Actual int8 GEMM vs independent CPU int32 accumulation; no speed or output-equivalence claim')
