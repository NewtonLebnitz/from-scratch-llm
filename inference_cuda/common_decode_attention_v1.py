"""Inference-only grouped single-query attention; direct K/V head access, no repetition."""
import torch,triton
import triton.language as tl

@triton.jit
def _decode(Q,K,V,M,O,QH:tl.constexpr,KH:tl.constexpr,KS:tl.constexpr,VH:tl.constexpr,VS:tl.constexpr,OH:tl.constexpr,D:tl.constexpr,N:tl.constexpr,GROUP:tl.constexpr,B:tl.constexpr):
 h=tl.program_id(0);kh=h//GROUP;t=tl.arange(0,B);d=tl.arange(0,D)
 valid=(t<N)&tl.load(M+t,mask=t<N,other=0)
 q=tl.load(Q+h*QH+d).to(tl.float32)
 k=tl.load(K+kh*KH+t[:,None]*KS+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
 scores=tl.sum(k*q[None,:],1)*(D**-0.5);scores=tl.where(valid,scores,float('-inf'))
 p=tl.exp(scores-tl.max(scores,0));den=tl.sum(p,0)
 v=tl.load(V+kh*VH+t[:,None]*VS+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
 out=tl.sum(p[:,None]*v,0)/den
 tl.store(O+h*OH+d,out)

def decode_attention(q,k,v,mask,warps=4):
 assert q.shape[0]==1 and q.shape[2]==1 and q.shape[-1] in [32,64,128]
 n=k.shape[2];out=torch.empty_like(q);d=q.shape[-1];groups=q.shape[1]//k.shape[1]
 _decode[(q.shape[1],)](q,k,v,mask,out,q.stride(1),k.stride(1),k.stride(2),v.stride(1),v.stride(2),out.stride(1),d,n,groups,triton.next_power_of_2(n),num_warps=max(warps,triton.next_power_of_2(max(1,n//64))))
 return out
