"""Task-scoped source/SOM retrieval; verified raw parents, no QA lookup."""
import hashlib
from pathlib import Path
"""Source-aware lexical/SOM candidate retrieval. No question-to-answer mappings.
Explicit source aliases apply to all documents, not selected diagnostic questions.
A small disclosed bilingual terminology glossary expands ranking queries only.
"""
import collections,json,math,re,unicodedata
import numpy as np
STOP=set('the a an what which how are is does do of in to and for its this that with from by on as it name give list used use why can into or please'.split())
def normalized(s):return re.sub(r'[^a-z0-9+\u4e00-\u9fff]','',unicodedata.normalize('NFKC',s).lower())
def raw_terms(text):
 out=[]
 for w in re.findall(r'[a-z0-9]+(?:\+\+)?|[\u4e00-\u9fff]+',unicodedata.normalize('NFKC',text).lower()):
  if re.fullmatch(r'[\u4e00-\u9fff]+',w):out.extend(w[i:i+2] for i in range(len(w)-1))
  elif w not in STOP and (len(w)>1 or w.isdigit()):out.append(w)
 return out

def terms(text):
 text=unicodedata.normalize('NFKC',text).replace('\u00ad','')
 text=re.sub(r'(?<=[A-Za-z])-\s+(?=[a-z])','',text)
 text=re.sub(r'\b([A-Z]{2,})\s+([A-Z]+s?)\b',lambda m:m[1]+m[2],text)
 text=re.sub(r'\b([A-Z]{2,})\s+(s)\b',lambda m:m[1]+m[2],text)
 result=raw_terms(text)
 # Conventional terminology equivalence in ranking only, never synthesized evidence.
 acronyms={'svr':'support vector regression regressor','svm':'support vector machine','pca':'principal component analysis','nmf':'nonnegative matrix factorization','paa':'piecewise aggregate approximation','sax':'symbolic aggregate approximation','vla':'vision language action'}
 expanded=[]
 for word in result:
  canonical=word[:-1] if word.endswith('s') and word[:-1] in acronyms else word
  expanded.append(canonical)
  if canonical in acronyms:expanded.extend(acronyms[canonical].split())
 result=expanded
 return [x[:-3] if len(x)>6 and x.endswith('ing') else x[:-2] if len(x)>5 and x.endswith('ed') else x[:-1] if len(x)>4 and x.endswith('s') else x for x in result]

class SourceAwareSOM:
 def __init__(self,index,labels,lexical_weight=2.):
  self.lexical_weight=lexical_weight;assert 0<=lexical_weight<=4
  self.index=index;self.labels=labels;self.n=len(index.documents);assert self.n<=50000
  self.postings=collections.defaultdict(list);length=[]
  for i,d in enumerate(index.documents):
   t=terms(d['title']+' '+d['text']);c=collections.Counter(t);length.append(len(t))
   for w,f in c.items():self.postings[w].append((i,f))
  self.lengths=np.array(length);self.average=max(float(self.lengths.mean()),1)
 def search(self,question,vector,*,top_k=2,nearest_nodes=8):
  assert 1<=top_k<=20
  q=normalized(question);matches=[]
  for sid,aliases in self.labels['sources'].items():
   for alias in aliases:
    a=normalized(alias)
    if a and a in q:matches.append((sid,a))
  # Longest nested alias wins; multiple non-nested source names are retained.
  matches=[(sid,a) for sid,a in matches if not any(a!=b and a in b for _,b in matches)]
  sources={sid for sid,_ in matches};dense=self.index.search(vector,top_k=20,nearest_nodes=nearest_nodes);a=self.index.arrays;docs=self.index.documents
  if sources:candidates=np.array([i for i,d in enumerate(docs) if d['source_id'] in sources],dtype=int)
  else:candidates=np.flatnonzero(np.isin(a['bmu'],dense['selected_nodes']))
  assert len(candidates)>0
  expanded=question+' '+' '.join(v for k,v in self.labels['query_glossary'].items() if k in question)
  lexical=np.zeros(self.n)
  query_terms=set(terms(expanded))
  # Once source identity is fixed, its repeated name is not evidence of local relevance.
  if len(sources)==1:
   sid=next(iter(sources));entity_terms=set(terms(' '.join(self.labels['sources'][sid])));query_terms-=entity_terms
  for term in query_terms:
   postings=self.postings.get(term,[]);df=len(postings)
   if not df:continue
   ix=np.array([i for i,f in postings]);tf=np.array([f for i,f in postings]);lexical[ix]+=math.log(1+(self.n-df+.5)/(df+.5))*tf*2.2/(tf+1.2*(.25+.75*self.lengths[ix]/self.average))
  cos=np.asarray(self.index.embeddings[candidates]@vector);den=candidates[np.argsort(-cos,kind='stable')].tolist();lex=candidates[np.argsort(-lexical[candidates],kind='stable')].tolist();rrf=collections.defaultdict(float)
  for weight,order in [(1.,den),(self.lexical_weight,lex)]:
   for rank,i in enumerate(order):rrf[int(i)]+=weight/(20+rank+1)
  ordered=sorted(rrf,key=lambda i:(-rrf[i],-float(self.index.embeddings[i]@vector),i));ranked=[];seen=set()
  for i in ordered:
   identity=docs[i].get('excerpt_id',docs[i].get('parent_chunk_id',docs[i]['id']))
   if identity in seen:continue
   seen.add(identity);ranked.append(i)
   if len(ranked)==top_k:break
  width=self.index.manifest['settings']['width'];results=[]
  for i in ranked:
   node=int(a['bmu'][i]);results.append(dict(docs[i],score=float(self.index.embeddings[i]@vector),hybrid_score=rrf[i],bmu=node,x=node%width,y=node//width,quantization_error=float(a['quantization_errors'][i])))
  return {**dense,'results':results,'candidate_count':len(candidates),'source_name_matches':sorted(sources),'query_expanded_for_ranking':expanded,'retrieval_method':'Explicit source-title disambiguation + lexical/SOM RRF20; no generated answer or per-question routing'}

class ParentSOM:
 def __init__(self,root,index,labels,lexical_weight=2.):
  self.root=Path(root).resolve();self.index=index
  # Independent module object uses normalization only for lexical ranking; raw spans remain intact.
  self.hybrid=SourceAwareSOM(index,labels,lexical_weight=lexical_weight)
  self.parents={}
  for doc in index.documents:
   path=(self.root/doc['reviewed_manifest']).resolve()
   if not path.is_relative_to(self.root/'datasets/manifests') or not path.is_file() or path.stat().st_size>8*1024**2:raise ValueError('Invalid reviewed source manifest')
   if str(path) not in self.parents:
    self.parents[str(path)]={x['id']:x for x in json.loads(path.read_text('utf-8-sig'))['excerpts']}
 def search(self,question,vector,*,top_k=2,nearest_nodes=1):
  ret=self.hybrid.search(question,vector,top_k=top_k,nearest_nodes=nearest_nodes)
  output=[]
  for row in ret['results']:
   parent=self.parents[str(self.root/row['reviewed_manifest'])][row['excerpt_id']]
   text=parent['text'];digest=hashlib.sha256(text.encode()).hexdigest()
   assert digest==parent['text_sha256']==row['original_excerpt_sha256']
   # Include immediately contiguous reviewed paragraphs on the same source page.
   # This is a document-position rule, not a question/answer routing rule.
   allowed={x['excerpt_id'] for x in self.index.documents if x['source_id']==row['source_id']}
   same=[x for x in self.parents[str(self.root/row['reviewed_manifest'])].values() if x['id'] in allowed and x['source_id']==row['source_id'] and x['page']==parent['page']]
   group=[parent];changed=True
   while changed:
    changed=False;lo=min(x['normalized_start'] for x in group);hi=max(x['normalized_end'] for x in group)
    for other in same:
     if other in group:continue
     if 0<=lo-other['normalized_end']<=2 or 0<=other['normalized_start']-hi<=2:group.append(other);changed=True
   group.sort(key=lambda x:x['normalized_start']);parts=[]
   for part in group:
    assert hashlib.sha256(part['text'].encode()).hexdigest()==part['text_sha256']
    parts.append(part['text'])
   text='\n\n'.join(parts);digest=hashlib.sha256(text.encode()).hexdigest()
   output.append(dict(row,text=text,sha256=digest,expanded_parent_excerpt=True,parent_page=parent['page'],retrieved_child_id=row['id'],expanded_excerpt_ids=[x['id'] for x in group],original_fragment_sha256=[x['text_sha256'] for x in group],adjacency_rule='Same source/page, reviewed normalized spans at most 2 characters apart; only TRAIN-cited excerpts'))
  return dict(ret,results=output,retrieval_method='Source-name disambiguation + SOM candidate expansion + normalized lexical(0.5)/dense(1) RRF; full hash-verified original excerpt expansion, no QA lookup',small_index_caveat='Source-named candidates include all excerpts of that source; no SOM quality/speed benefit claim for this tiny collection')


def compact_sources(question,records,tokenizer,system,*,max_prompt_tokens=640):
 from .data import conversation_prefix,clean_document
 parts=[];included=[];omitted=[]
 for row in records:
  text=clean_document(row['text']);prefix=''  # Training evidence is raw prose; citations remain in separate references.
  full='\n\n'.join(parts+[prefix+text]);clipped=False
  if len(tokenizer.encode(conversation_prefix(question,full,system)))>max_prompt_tokens:
   lo=0;hi=len(text)
   while lo<hi:
    mid=(lo+hi+1)//2;candidate='\n\n'.join(parts+[prefix+text[:mid]])
    if len(tokenizer.encode(conversation_prefix(question,candidate,system)))<=max_prompt_tokens:lo=mid
    else:hi=mid-1
   if lo<80:omitted.append(row['id']);continue
   text=text[:lo];clipped=True
  parts.append(prefix+text);included.append(dict(row,text=text,clipped_for_context=clipped,prompt_reference_number=len(included)+1))
 context='\n\n'.join(parts);assert len(tokenizer.encode(conversation_prefix(question,context,system)))<=max_prompt_tokens
 return context,included,omitted


def retrieve_sources(root,index_name,encoder,question,*,top_k=4,nearest_nodes=8):
 """Route by versioned index profile, never by a question-to-answer map."""
 from .retrieval import SOMIndex
 root=Path(root).resolve();vector=encoder.encode([question],query=True)[0]
 with SOMIndex(root,root/'artifacts/som'/index_name,encoder.identity) as index:
  profile=index.manifest.get('retrieval_profile')
  if not profile:return index.search(vector,top_k=top_k,nearest_nodes=nearest_nodes)
  if profile.get('format')!='source-parent-som-halflex-r4':raise ValueError('Unknown retrieval profile')
  if profile.get('lexical_weight')!=.5 or profile.get('max_prompt_tokens')!=722:raise ValueError('Profile contract changed')
  path=(root/profile['labels_path']).resolve()
  if not path.is_relative_to(root/'configs') or hashlib.sha256(path.read_bytes()).hexdigest()!=profile['labels_sha256']:raise ValueError('Retrieval labels changed')
  labels=json.loads(path.read_text('utf-8'));q=normalized(question)
  matches=[(sid,normalized(alias)) for sid,aliases in labels['sources'].items() for alias in aliases if normalized(alias) and normalized(alias) in q]
  matches=[(sid,a) for sid,a in matches if not any(a!=b and a in b for _,b in matches)]
  unavailable={sid for sid,_ in matches}-{d['source_id'] for d in index.documents}
  if unavailable:raise ValueError('Selected index has no source documents for: '+', '.join(sorted(unavailable)))
  ret=ParentSOM(root,index,labels,lexical_weight=.5).search(question,vector,top_k=top_k,nearest_nodes=nearest_nodes)
  return dict(ret,prompt_token_budget=722,profile=profile,validated_scope=index.manifest.get('validated_scope'),matches_validated_retrieval_parameters=(top_k==1 and nearest_nodes==1))
