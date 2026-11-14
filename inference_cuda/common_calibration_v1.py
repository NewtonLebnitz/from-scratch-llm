"""TRAIN-only bounded calibration for custom-model AWQ/GPTQ/FP8 inference ports."""
import collections,gc,hashlib,json,time,torch
from mlsystem.components import Linear
from mlsystem.genlora import merge
from mlsystem.ols_rbf import RBFAdaptedModel
from mlsystem.data import conversation_prefix
from common_quant_principles_v1 import quantize_groups,awq_calibrate,gptq_calibrate,PackedInt4Linear,FP8Linear,W8A8Linear

def select_train(root,tok):
 path=root/'datasets/processed/expanded-qa64-v6d/train.json';records=json.loads(path.read_text())['records'];groups=collections.defaultdict(list)
 for r in records:
  if r['condition']=='without_evidence':groups[r['fact_id']].append(r)
 selected=[]
 for i,(fact,rows)in enumerate(sorted(groups.items())):
  language=['zh','en'][i%2];choices=[r for r in rows if r['language']==language]
  row=min(choices,key=lambda r:hashlib.sha256(r['question'].encode()).hexdigest())
  ids=tok.encode(conversation_prefix(row['question'],row['context'],row['system'])+row['answer']+'<|turn_end|>')
  assert 0<len(ids)<=512;selected.append(dict(id=row['id'],fact_id=fact,language=language,ids=ids))
 assert len(selected)==80
 return selected

@torch.no_grad()
def calibrate_and_convert(model,method,root,tok):
 started=time.perf_counter();base=model.base if isinstance(model,RBFAdaptedModel)else model
 merge(base);base.half()
 if isinstance(model,RBFAdaptedModel):model.adapter.float()
 targets=[(n,m)for n,m in base.named_modules()if isinstance(m,Linear)];stats={};hooks=[]
 if method in ['awq_int4','gptq_int4','smoothquant_int8']:
  selected=select_train(root,tok)
  for name,module in targets:
   width=module.weight.shape[1]
   state=dict(rows=0,samples=[],sample_rows=0)
   if method=='gptq_int4':state['gram']=torch.zeros((width,width),device='cuda',dtype=torch.float32)
   if method=='smoothquant_int8':state['input_max']=torch.zeros(width,device='cuda')
   stats[name]=state
   def before(m,args,state=state):
    x=args[0].detach().reshape(-1,args[0].shape[-1]).float()
    if 'gram'in state:state['gram'].add_(x.T@x)
    if 'input_max'in state:state['input_max']=torch.maximum(state['input_max'],x.abs().amax(0))
    state['rows']+=x.shape[0]
    positions=torch.tensor([x.shape[0]//3,2*x.shape[0]//3],device=x.device)
    state['samples'].append(x.index_select(0,positions).clone());state['sample_rows']+=2
   hooks.append(module.register_forward_pre_hook(before))
  try:
   for row in selected:base(torch.tensor([row['ids']],device='cuda',dtype=torch.long))
  finally:
   for h in hooks:h.remove()
 else:selected=[]
 report=[]
 for name,module in targets:
  if method=='fp8_scaled_mm':q=FP8Linear(module.weight);info=dict(method='PyTorch actual scaled FP8 GEMM, per-tensor dynamic activation scale, E4M3; padded rows to16')
  elif method in ['int8_rtn','smoothquant_int8']:
   state=stats.pop(name,None);q=W8A8Linear(module.weight,state['input_max']if state else None)
   info=dict(method='Actual torch INT8xINT8->INT32 GEMM; dynamic per-token symmetric activation scaling; per-output weight scaling',smoothquant_alpha=.5 if state else None,
    activation_input_rescaling=bool(state),principle_port=True)
   if state:info['calibration_token_positions']=state['rows']
   del state
  else:
   state=stats.pop(name,None)
   if method=='awq_int4':
    samples=torch.cat(state['samples']);indexes=torch.linspace(0,len(samples)-1,min(64,len(samples)),device=samples.device).long()
    integers,scales,inputs,info=awq_calibrate(module.weight,samples.index_select(0,indexes))
   elif method=='gptq_int4':integers,scales,inputs,info=gptq_calibrate(module.weight,2*state['gram']/state['rows'])
   elif method=='rtn_int4':
    integers,scales,_=quantize_groups(module.weight);inputs=torch.ones(module.weight.shape[1],device='cuda');info=dict(method='Symmetric per-output group128 round-to-nearest INT4')
   else:raise ValueError(method)
   q=PackedInt4Linear(integers,scales,inputs)
   if state:info['calibration_token_positions']=state['rows']
   del state,integers,scales,inputs
  parent,_,leaf=name.rpartition('.');setattr(base.get_submodule(parent)if parent else base,leaf,q)
  report.append(dict(layer=name,**info))
 del stats;gc.collect();torch.cuda.synchronize()
 return dict(method=method,seconds=time.perf_counter()-started,calibration_train_ids=[r['id']for r in selected],calibration_sequences=len(selected),
  calibration_fact_groups=len({r['fact_id']for r in selected}),layers=report,official_package=False,weights_saved=False,training_modified=False,
  distinction='Mechanism-level port to handwritten Linear; actual packed INT4 Triton GEMM or actual torch scaled FP8 kernel. Not a claim to reproduce every default in AutoAWQ/GPTQModel.')
