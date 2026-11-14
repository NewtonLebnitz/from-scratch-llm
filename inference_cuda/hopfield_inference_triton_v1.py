"""Single-query Hopfield score-EMA, softmax and grouped V principle port.
Returns every next-layer score; never replaces Hopfield with ordinary attention.
"""
import torch,triton,triton.language as tl
@triton.jit
def _hop(Q,K,V,PREV,OUT,STATE,POS,TABLE,KS,VS,
 Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
 K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
 V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
 S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,
 H:tl.constexpr,HK:tl.constexpr,N:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr,
 HAS_PREV:tl.constexpr,ALPHA:tl.constexpr,PAGED:tl.constexpr,PAGE:tl.constexpr,QUANT:tl.constexpr):
 h=tl.program_id(0);b=tl.program_id(1);kh=h//(H//HK);t=tl.arange(0,BLOCK);d=tl.arange(0,D)
 physical=t
 if PAGED:physical=tl.load(TABLE+t//PAGE,t<N,0)*PAGE+t%PAGE
 q=tl.load(Q+b*Q0+h*Q1+d*Q3).to(tl.float32)
 k=tl.load(K+b*K0+kh*K1+physical[:,None]*K2+d[None,:]*K3,t[:,None]<N,0).to(tl.float32)
 v=tl.load(V+b*V0+kh*V1+physical[:,None]*V2+d[None,:]*V3,t[:,None]<N,0).to(tl.float32)
 if QUANT:
  sk=tl.load(KS+b*S0+kh*S1+physical*S2,t<N,1);sv=tl.load(VS+b*S0+kh*S1+physical*S2,t<N,1)
  k=k*sk[:,None];v=v*sv[:,None]
 scores=tl.sum(k*q[None,:],1)
 # Match eager half/BF16 matmul's output rounding before float scaling.
 scores=scores.to(Q.dtype.element_ty).to(tl.float32)*(D**-0.5)
 valid=(t<N)&(t<=tl.load(POS));scores=tl.where(valid,scores,0.)
 if HAS_PREV:
  old=tl.load(PREV+(b*H+h)*N+t,t<N,0);scores=ALPHA*old+(1.-ALPHA)*scores
 tl.store(STATE+(b*H+h)*N+t,scores,t<N)
 z=tl.where(valid,scores,-float('inf'));p=tl.exp(z-tl.max(z,0));p=p/tl.sum(p,0)
 p=p.to(Q.dtype.element_ty).to(tl.float32)
 result=tl.sum(p[:,None]*v,0);tl.store(OUT+(b*H+h)*D+d,result)

def hopfield_decode(q,k,v,mask,previous,alpha,table=None,page=16,ks=None,vs=None):
 assert q.shape[-2]==1 and q.shape[-1]==64
 if previous is not None:previous=previous.contiguous()
 n=mask.shape[-1];b,h,_,d=q.shape;state=torch.empty((b,h,1,n),device=q.device,dtype=torch.float32);out=torch.empty_like(q)
 # The mask row contains the causal position, no data-dependent CPU read.
 pos=(mask.long().sum(-1)-1).to(torch.long)
 scales=ks.stride()if ks is not None else(0,0,0)
 _hop[(h,b)](q,k,v,previous if previous is not None else state,out,state,pos,table if table is not None else pos,ks if ks is not None else pos,vs if vs is not None else pos,
  *q.stride(),*k.stride(),*v.stride(),*scales,h,k.shape[1],n,d,triton.next_power_of_2(n),previous is not None,float(alpha),table is not None,page,ks is not None,num_warps=4)
 return out,state
