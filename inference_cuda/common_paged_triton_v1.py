"""Single-query GQA attention reading physical pages directly; no canonical KV gather."""
import torch,triton
import triton.language as tl

@triton.jit
def _paged(Q,K,V,P,Y,L,C:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,PAGE:tl.constexpr,T:tl.constexpr):
 head=tl.program_id(0);kv_head=head//(HQ//HK);t=tl.arange(0,T);d=tl.arange(0,D)
 page=tl.load(P+t//PAGE,mask=t<C,other=0).to(tl.int32)
 physical=page*PAGE+t%PAGE
 q=tl.load(Q+head*D+d).to(tl.float32)
 k=tl.load(K+kv_head*C*D+physical[:,None]*D+d[None,:],mask=(t[:,None]<L)&(t[:,None]<C),other=0).to(tl.float32)
 scores=tl.sum(k*q[None,:],1)*(D**-.5);scores=tl.where(t<L,scores,float('-inf'))
 p=tl.exp(scores-tl.max(scores,0));p=p/tl.sum(p,0)
 v=tl.load(V+kv_head*C*D+physical[:,None]*D+d[None,:],mask=(t[:,None]<L)&(t[:,None]<C),other=0).to(tl.float32)
 value=tl.sum(v*p[:,None],0);tl.store(Y+head*D+d,value)

def paged_decode(q,k,v,table,length,page_size):
 assert q.shape[0]==1 and q.shape[2]==1
 q=q.contiguous();out=torch.empty_like(q);c=k.shape[2];d=q.shape[-1];assert 0<length<=c
 _paged[(q.shape[1],)](q,k,v,table,out,length,c,d,q.shape[1],k.shape[1],page_size,max(16,triton.next_power_of_2(length)),num_warps=8)
 return out
