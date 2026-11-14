"""Greedy exact-verification prompt lookup and layer-skip draft speculation."""
import copy,time,torch
from torch import nn
from mlsystem.ols_rbf import RBFAdaptedModel
from common_methods_v8_engine import Engine
from common_cache_principles_v1 import VerificationEngine

def prompt_proposal(sequence,limit):
 for width in [4,3,2]:
  if len(sequence)<=width:continue
  suffix=sequence[-width:]
  for start in range(len(sequence)-width-1,-1,-1):
   if sequence[start:start+width]==suffix:
    out=sequence[start+width:start+width+limit]
    if out:return out
 return []

class SpeculativeEngine:
 def __init__(self,model,spec):
  self.model=model;self.spec=spec;target_spec={**spec,'cache':'static','attention':'manual'}
  self.target=VerificationEngine(model,target_spec);self.base=self.target.base;self.question=self.target.question;self.draft=None
  if spec['speculation']=='layer_skip':
   # Only the ephemeral draft changes depth. Target model and its modules are untouched.
   partial=copy.copy(self.base);partial._modules=self.base._modules.copy()
   partial.layers=nn.ModuleList(list(self.base.layers)[:len(self.base.layers)//2])
   draft_model=RBFAdaptedModel(partial,model.adapter)if isinstance(model,RBFAdaptedModel)else partial
   draft_model.eval();self.draft=Engine(draft_model,target_spec)
  self.draft_depth=len(self.draft.base.layers)if self.draft else 0
 def prepare_acceleration(self):return dict(seconds=0)

 def generate(self,tok,case,settings,banned,stops):
  torch.cuda.synchronize();start=time.perf_counter();generated=[];first=None;reason='length_limit';termination=None;question_seconds=0
  q=None
  if self.question is not None:
   t=time.perf_counter();q=self.model.adapter.encode_question(tok,case['question']);torch.cuda.synchronize();question_seconds=time.perf_counter()-t
  self.target.reset(q)
  if self.draft:self.draft.reset(q)
  prompt=case['prompt_ids'];limit=settings['max_new_tokens'];calls=0;proposed=0;accepted=0
  def choose(z):
   z=z.float()
   if not torch.isfinite(z).all():raise RuntimeError('Nonfinite speculative target/draft logits')
   z[...,banned]=-torch.inf
   return z.argmax(-1)
  token=int(choose(self.target.last_logits(torch.tensor([prompt],device='cuda')))[0]);calls+=1
  if token in stops:reason='model_terminator';termination=token
  else:generated.append(token);first=time.perf_counter()-start
  while generated and len(generated)<limit and reason!='model_terminator':
   capacity=min(4,limit-len(generated)-1);old_position=self.target.position
   if self.draft is None:proposal=prompt_proposal(prompt+generated,capacity)if capacity else []
   else:
    prefix=prompt+generated[:-1];desired=len(prefix)
    if self.draft.position>desired:self.draft.position=desired
    if self.draft.position<desired:
     self.draft.last_logits(torch.tensor([prefix[self.draft.position:]],device='cuda'))
    proposal=[];last=generated[-1]
    for _ in range(capacity):
     last=int(choose(self.draft.last_logits(torch.tensor([[last]],device='cuda')))[0]);proposal.append(last)
     if last in stops:break
   # Verify every proposal, plus one bonus/correction, against the full target.
   all_logits=self.target.all_logits(torch.tensor([[generated[-1]]+proposal],device='cuda'))
   predictions=choose(all_logits)[0].tolist();calls+=1;proposed+=len(proposal)
   match=0
   for prediction,draft_token in zip(predictions,proposal):
    if prediction!=draft_token:break
    match+=1
   accepted+=match;output=predictions[:match+1]
   appended=0
   for token in output:
    if token in stops:reason='model_terminator';termination=token;break
    generated.append(token);appended+=1
    if len(generated)>=limit:break
   # Only accepted-prefix KV positions remain logically valid; unused physical slots are masked.
   self.target.position=old_position+appended
  torch.cuda.synchronize();seconds=time.perf_counter()-start;decode=seconds-first if first is not None else None
  return dict(token_ids=generated,text=tok.decode(generated),stop_reason=reason,termination_token=termination,
   prompt_tokens=len(prompt),tokens=len(generated),model_steps=calls,seconds=seconds,first_token_seconds=first,
   decode_tokens_per_second=(len(generated)-1)/decode if decode and len(generated)>1 else None,total_tokens_per_second=len(generated)/seconds,
   question_setup_seconds=question_seconds,verification_calls=calls,draft_tokens=proposed,accepted_draft_tokens=accepted,
   acceptance_rate=accepted/proposed if proposed else None,draft_depth=self.draft_depth)
