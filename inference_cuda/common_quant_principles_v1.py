"""TRAIN-calibrated AWQ/GPTQ principles and actual packed INT4/FP8 execution.

These are mechanism-level ports for handwritten Linear modules, not claims of
bit-identical AWQ/GPTQ package implementations. No retraining or model publishing.
"""
import math
import torch
from torch import nn

def quantize_groups(weight,group=128):
 assert weight.ndim==2 and weight.shape[1]%group==0
 w=weight.float().reshape(weight.shape[0],-1,group)
 scale=w.abs().amax(-1).clamp_min(1e-6)/7
 integers=(w/scale[...,None]).round().clamp(-8,7).to(torch.int8).reshape_as(weight)
 dequant=(integers.reshape_as(w).float()*scale[...,None]).reshape_as(weight)
 return integers,scale,dequant

def pack_int4(integers):
 assert integers.dtype==torch.int8 and integers.shape[-1]%2==0
 unsigned=(integers.to(torch.int16)+8).to(torch.uint8)
 return unsigned[:,0::2]|(unsigned[:,1::2]<<4)

def unpack_int4(packed):
 values=torch.stack([packed&15,packed>>4],-1).flatten(-2)
 return(values.to(torch.int16)-8).to(torch.int8)

def awq_calibrate(weight,inputs,group=128):
 x=inputs.float();w=weight.float();act=x.abs().mean(0).clamp_min(1e-6);mean_w=w.abs().mean(0).clamp_min(1e-6)
 target=x@w.T;best=None
 for alpha in [None,0.,.25,.5,.75,1.]:
  s=torch.ones_like(act)if alpha is None else(act.pow(alpha)/mean_w.pow(1-alpha)).clamp(1e-3,1e3)
  s=s/(s.max()*s.min()).sqrt()
  q,scale,dq=quantize_groups(w*s[None],group)
  error=float(((x/s[None])@dq.T-target).square().mean())
  if best is None or error<best[0]:best=(error,alpha,q,scale,s)
 error,alpha,q,scale,s=best
 return q,scale,s,dict(method='Activation-aware scaling and TRAIN reconstruction-error search; symmetric group INT4; principle port, not official AutoAWQ',alpha=alpha,calibration_mse=error)

def gptq_calibrate(weight,hessian,group=128,damp=.01):
 # Full input covariance, exact Cholesky inverse; blockwise sequential error feedback.
 w=weight.float().clone();H=hessian.float().clone();n=w.shape[1];assert n%group==0 and H.shape==(n,n)
 diagonal=torch.diagonal(H);dead=diagonal==0;H[dead,dead]=1;w[:,dead]=0
 H.diagonal().add_(damp*H.diagonal().mean())
 inverse=torch.cholesky_inverse(torch.linalg.cholesky(H))
 upper=torch.linalg.cholesky(inverse,upper=True)
 integers=torch.empty_like(w,dtype=torch.int8);scales=torch.empty((w.shape[0],n//group),device=w.device)
 for start in range(0,n,group):
  end=start+group;block=w[:,start:end].clone();errors=torch.zeros_like(block);U=upper[start:end,start:end]
  scale=block.abs().amax(-1).clamp_min(1e-6)/7;scales[:,start//group]=scale
  for column in range(group):
   original=block[:,column].clone();q=(original/scale).round().clamp(-8,7)
   integers[:,start+column]=q.to(torch.int8);error=(original-q*scale)/U[column,column]
   errors[:,column]=error
   block[:,column:]-=error[:,None]*U[column,column:][None,:]
  if end<n:w[:,end:]-=errors@upper[start:end,end:]
 return integers,scales,torch.ones(n,device=w.device),dict(method='Full-covariance damped GPTQ sequential error feedback, block/group128, no act-order; principle port, not official GPTQModel',damp=damp,dead_inputs=int(dead.sum()))

class PackedInt4Linear(nn.Module):
 def __init__(self,integers,scales,input_scale):
  super().__init__();self.in_features=integers.shape[1];self.out_features=integers.shape[0]
  self.register_buffer('packed',pack_int4(integers).contiguous())
  self.register_buffer('scales',scales.float().contiguous());self.register_buffer('input_scale',input_scale.float().contiguous())
 def forward(self,x):
  from common_quant_triton_v1 import int4_linear
  return int4_linear(x,self.packed,self.scales,self.input_scale)

class FP8Linear(nn.Module):
 def __init__(self,weight):
  super().__init__();scale=weight.float().abs().max().clamp_min(1e-12)/torch.finfo(torch.float8_e4m3fn).max
  self.register_buffer('scale',scale.float());self.register_buffer('weight8',(weight/scale).to(torch.float8_e4m3fn).contiguous())
 def forward(self,x):
  shape=x.shape;flat=x.reshape(-1,shape[-1]);rows=flat.shape[0]
  if rows%16:flat=torch.nn.functional.pad(flat,(0,0,0,16-rows%16))
  scale=flat.float().abs().max().clamp_min(1e-12)/torch.finfo(torch.float8_e4m3fn).max
  values=(flat/scale).to(torch.float8_e4m3fn)
  out=torch._scaled_mm(values,self.weight8.T,scale_a=scale.float(),scale_b=self.scale,out_dtype=x.dtype)
  return out[:rows].reshape(*shape[:-1],self.weight8.shape[0])

class W8A8Linear(nn.Module):
 """Actual int8 x int8 -> int32 GEMM, rowwise dynamic activation scaling."""
 def __init__(self,weight,activation_max=None,alpha=.5):
  super().__init__();w=weight.float()
  inputs=torch.ones(w.shape[1],device=w.device)
  if activation_max is not None:
   inputs=(activation_max.float().clamp_min(1e-5).pow(alpha)/w.abs().amax(0).clamp_min(1e-5).pow(1-alpha)).clamp(1e-4,1e4)
   inputs=inputs/(inputs.max()*inputs.min()).sqrt()
  transformed=w*inputs[None];scales=transformed.abs().amax(-1).clamp_min(1e-8)/127
  self.register_buffer('weight8',(transformed/scales[:,None]).round().clamp(-127,127).to(torch.int8).contiguous())
  self.register_buffer('scales',scales);self.register_buffer('input_scale',inputs)
 def forward(self,x):
  shape=x.shape;flat=x.reshape(-1,shape[-1]).float()/self.input_scale;rows=len(flat)
  if rows%16:flat=torch.nn.functional.pad(flat,(0,0,0,16-rows%16))
  sx=flat.abs().amax(-1).clamp_min(1e-8)/127;values=(flat/sx[:,None]).round().clamp(-127,127).to(torch.int8)
  result=torch._int_mm(values,self.weight8.T).float()*sx[:,None]*self.scales[None]
  return result[:rows].to(x.dtype).reshape(*shape[:-1],self.weight8.shape[0])
