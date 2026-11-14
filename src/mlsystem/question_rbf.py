"""Explicit question-conditioned Gaussian logit residual; no stored answers."""
import torch
from mlsystem.ols_rbf import GaussianLogitAdapter
from mlsystem.data import clean_document
class QuestionGaussianAdapter(GaussianLogitAdapter):
 def __init__(self,mean,projection,feature_scale,centers,widths,readout):
  super().__init__(mean,projection,feature_scale,centers,widths,readout)
  assert mean.numel()%2==0
  self.hidden_width=mean.numel()//2
  self.register_buffer('_question_mean',torch.empty(0,device=mean.device),persistent=False)
 @torch.no_grad()
 def set_question(self,base,tokenizer,question):
  ids=tokenizer.encode(clean_document(question));assert ids
  value=base.token_embeddings(torch.tensor(ids,device=self.mean.device)).float().mean(0)
  if value.shape!=(self.hidden_width,):raise ValueError('Question embedding width mismatch')
  self._question_mean=value.detach()
 def clear_question(self):self._question_mean=torch.empty(0,device=self.mean.device)
 def forward(self,hidden):
  if self._question_mean.shape!=(self.hidden_width,):raise RuntimeError('Explicit current question required; no stale condition reused')
  if hidden.shape[-1]!=self.hidden_width:raise ValueError('Frozen hidden width mismatch')
  condition=self._question_mean.expand(*hidden.shape[:-1],self.hidden_width)
  return super().forward(torch.cat([hidden.float(),condition],-1))

def make_question_adapter(state, *, hidden_width, vocabulary_size):
 required={'mean','projection','feature_scale','centers','raw_widths','readout'}
 if set(state)!=required:raise ValueError('Question Gaussian parameter keys disagree')
 mean,P,scale,C,rho,W=[state[k] for k in ['mean','projection','feature_scale','centers','raw_widths','readout']]
 if mean.shape!=(2*hidden_width,) or P.ndim!=2 or P.shape[0]!=2*hidden_width or not 1<=P.shape[1]<=384:raise ValueError('Question Gaussian projection shape invalid')
 p=P.shape[1]
 if C.ndim!=2 or not 1<=C.shape[0]<=1024 or C.shape[1]!=p or scale.shape!=(p,) or rho.shape!=(C.shape[0],) or W.shape!=(C.shape[0],vocabulary_size):raise ValueError('Question Gaussian center/readout shape invalid')
 if any(not torch.isfinite(t).all() for t in [mean,P,scale,C,rho,W]) or not (scale>0).all():raise ValueError('Question Gaussian invalid numerical state')
 adapter=QuestionGaussianAdapter(mean,P,scale,C,torch.nn.functional.softplus(rho).clamp_min(1e-4),W)
 projection=adapter.projection.detach().clone();del adapter._buffers['projection'];adapter.register_parameter('projection',torch.nn.Parameter(projection))
 adapter.load_state_dict(state,strict=True);adapter.clear_question()
 return adapter
