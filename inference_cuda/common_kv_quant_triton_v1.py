"""Compressed INT8 K/V, per-token/head scales, fused dequantizing GQA decode."""
import torch,triton
import triton.language as tl

@triton.jit
def _kv8(Q,K,V,KS,VS,Y,L,C:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,T:tl.constexpr):
 head=tl.program_id(0);kh=head//(HQ//HK);t=tl.arange(0,T);d=tl.arange(0,D)
 q=tl.load(Q+head*D+d).to(tl.float32);ks=tl.load(KS+kh*C+t,mask=t<L,other=0)
 k=tl.load(K+kh*C*D+t[:,None]*D+d[None,:],mask=t[:,None]<L,other=0).to(tl.float32)*ks[:,None]
 scores=tl.sum(k*q[None],1)*(D**-.5);scores=tl.where(t<L,scores,float('-inf'))
 p=tl.exp(scores-tl.max(scores,0));p=p/tl.sum(p,0);vs=tl.load(VS+kh*C+t,mask=t<L,other=0)
 v=tl.load(V+kh*C*D+t[:,None]*D+d[None,:],mask=t[:,None]<L,other=0).to(tl.float32)*vs[:,None]
 out=tl.sum(v*p[:,None],0);tl.store(Y+head*D+d,out)

def kv8_decode(q,k,v,ks,vs,length):
 assert q.shape[0]==1 and q.shape[2]==1 and k.dtype==torch.int8 and v.dtype==torch.int8
 q=q.contiguous();out=torch.empty_like(q);c=k.shape[2];d=q.shape[-1];assert 0<length<=c
 _kv8[(q.shape[1],)](q,k,v,ks,vs,out,length,c,d,q.shape[1],k.shape[1],max(16,triton.next_power_of_2(length)),num_warps=8)
 return out
