"""Actual packed symmetric INT4 fused-dequant GEMM; no FP16 weight materialization."""
import torch
import triton
import triton.language as tl

@triton.jit
def _int4(A,W,S,XS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=32,BK:tl.constexpr=32):
 mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);ki=tl.arange(0,BK)
 acc=tl.full((BM,BN),0,tl.float32)
 for base in range(tl.cdiv(K,BK)):
  kk=base*BK+ki
  a=tl.load(A+mi[:,None]*K+kk[None,:],mask=(mi[:,None]<M)&(kk[None,:]<K),other=0).to(tl.float32)
  xs=tl.load(XS+kk,mask=kk<K,other=1);a=(a/xs[None,:]).to(tl.float16)
  packed=tl.load(W+ni[None,:]*(K//2)+(kk[:,None]//2),mask=(ni[None,:]<N)&(kk[:,None]<K),other=0)
  q=((packed>>((kk[:,None]%2)*4))&15).to(tl.int32)-8
  scale=tl.load(S+ni[None,:]*(K//G)+(kk[:,None]//G),mask=(ni[None,:]<N)&(kk[:,None]<K),other=0)
  b=(q.to(tl.float32)*scale).to(tl.float16)
  acc+=tl.dot(a,b)
 tl.store(Y+mi[:,None]*N+ni[None,:],acc,mask=(mi[:,None]<M)&(ni[None,:]<N))

def int4_linear(x,packed,scales,input_scale):
 assert x.dtype==torch.float16 and x.is_cuda
 original=x.shape;a=x.reshape(-1,x.shape[-1]).contiguous();m,k=a.shape;n=packed.shape[0]
 assert packed.shape[1]*2==k and k%128==0
 y=torch.empty((m,n),device=x.device,dtype=x.dtype)
 _int4[(triton.cdiv(m,16),triton.cdiv(n,32))](a,packed,scales,input_scale,y,m,n,k,128)
 return y.reshape(*original[:-1],n)
