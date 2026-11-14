"""TRAIN-only BPE unigram/bigram geometry for a real Gaussian autoregressive adapter.
Vectorizer stores token ngrams and unsupervised SVD buffers, never questions or answers.
"""
import collections,math
import numpy as np
import torch
from mlsystem.ols_rbf import GaussianLogitAdapter,gaussian,token_error_target
from mlsystem.data import clean_document

def grams(tokenizer,question):
    ids=tokenizer.encode(clean_document(question).lower())
    if not ids:raise ValueError('Empty question')
    return [(int(i),-1)for i in ids]+[(int(a),int(b))for a,b in zip(ids,ids[1:])]

def fit_lexical(tokenizer,questions,dimensions=256):
    # Unique TRAIN wording, not answer-length-weighted and not DEV/TEST vocabulary.
    questions=list(dict.fromkeys(questions));counts=[collections.Counter(grams(tokenizer,q))for q in questions]
    df=collections.Counter(k for c in counts for k in c);keys=sorted(df);lookup={k:i for i,k in enumerate(keys)}
    if not 1<=dimensions<min(len(questions),len(keys)):raise ValueError('Invalid lexical SVD budget')
    idf=np.array([math.log((1+len(questions))/(1+df[k]))+1 for k in keys],np.float32)
    X=np.zeros((len(counts),len(keys)),np.float32)
    for row,c in zip(X,counts):
        for key,n in c.items():row[lookup[key]]=n
    X*=idf;X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-6);mean=X.mean(0)
    _,singular,vt=np.linalg.svd(X-mean,full_matrices=False)
    axes=vt[:dimensions].T.copy();scale=np.sqrt(np.maximum(singular[:dimensions]/math.sqrt(len(X)-1),1e-3))
    values=(X-mean)@axes/scale
    rms=float(np.sqrt((values**2).mean()));scale*=max(rms,1e-6)
    buffers={'lex_keys':torch.tensor(keys,dtype=torch.int64),'lex_idf':torch.from_numpy(idf),'lex_mean':torch.from_numpy(mean),'lex_axes':torch.from_numpy(axes),'lex_scale':torch.from_numpy(scale)}
    return buffers,{'unique_train_wordings':len(questions),'ngram_count':len(keys),'dimensions':dimensions,'weighting':'raw count * TRAIN-only smoothed IDF, row L2','projection':'TRAIN-only centered SVD, inverse sqrt standard deviation, one global RMS normalization','answers_used_for_vectorizer':False,'dev_or_test_used_for_vectorizer':False}

class LexicalGaussianAdapter(GaussianLogitAdapter):
    def __init__(self,mean,projection,feature_scale,centers,widths,readout,lexical,hidden_width=768):
        super().__init__(mean,projection,feature_scale,centers,widths,readout)
        self.hidden_width=hidden_width;self.question_width=mean.numel()-hidden_width
        if self.question_width!=lexical['lex_axes'].shape[1]:raise ValueError('Lexical width disagreement')
        for k,v in lexical.items():self.register_buffer(k,v.detach().to(mean.device).clone())
        self._lex_lookup={tuple(k):i for i,k in enumerate(self.lex_keys.cpu().tolist())}
        self.register_buffer('_question_mean',torch.empty(0,device=mean.device),persistent=False)
    @torch.no_grad()
    def encode_question(self,tokenizer,question):
        x=torch.zeros(len(self.lex_keys),device=self.mean.device,dtype=torch.float32)
        for key,n in collections.Counter(grams(tokenizer,question)).items():
            i=self._lex_lookup.get(key)
            if i is not None:x[i]=n
        x=x*self.lex_idf;x=x/x.norm().clamp_min(1e-6)
        return ((x-self.lex_mean)@self.lex_axes)/self.lex_scale
    @torch.no_grad()
    def set_question(self,base,tokenizer,question):
        if base.config.d_model!=self.hidden_width:raise ValueError('Parent hidden width mismatch')
        self._question_mean=self.encode_question(tokenizer,question).detach()
    def clear_question(self):self._question_mean=torch.empty(0,device=self.mean.device)
    def forward(self,hidden):
        if self._question_mean.shape!=(self.question_width,):raise RuntimeError('Explicit current question required')
        if hidden.shape[-1]!=self.hidden_width:raise ValueError('Hidden width mismatch')
        return super().forward(torch.cat([hidden.float(),self._question_mean.expand(*hidden.shape[:-1],self.question_width)],-1))

def question_vector(lexical,tokenizer,question):
    keys=lexical['lex_keys'];lookup={tuple(k):i for i,k in enumerate(keys.cpu().tolist())};x=torch.zeros(len(keys),device=lexical['lex_idf'].device)
    for key,n in collections.Counter(grams(tokenizer,question)).items():
        if key in lookup:x[lookup[key]]=n
    x=x*lexical['lex_idf'];x=x/x.norm().clamp_min(1e-6)
    return ((x-lexical['lex_mean'])@lexical['lex_axes'])/lexical['lex_scale']

@torch.no_grad()
def prepare_lexical_candidates(hidden,base_logits,labels,*,candidate_rows,dimensions=320,candidates=1536,seed=7,width_mode='global',hidden_dimensions=64,question_dimensions=256):
    if hidden.shape[1]!=768+question_dimensions or dimensions!=hidden_dimensions+question_dimensions or len(candidate_rows)!=candidates or len(candidate_rows.unique())!=candidates:raise ValueError('Invalid candidate geometry')
    x=hidden[:,:768].float();mu=x.mean(0);_,sv,vt=torch.linalg.svd(x-mu,full_matrices=False)
    P=torch.zeros((hidden.shape[1],dimensions),device=hidden.device);P[:768,:hidden_dimensions]=vt[:hidden_dimensions].T;P[768:,hidden_dimensions:]=torch.eye(question_dimensions,device=hidden.device)
    q=hidden[:,768:].float();qm=q.mean(0);qrms=(q-qm).square().mean().sqrt().clamp_min(1e-4)
    mean=torch.cat([mu,qm]);scale=torch.cat([(sv[:hidden_dimensions]/math.sqrt(len(x)-1)).clamp_min(1e-3),qrms.repeat(question_dimensions)])
    f=(hidden-mean)@P/scale;centers=f[candidate_rows];d=torch.cdist(centers,centers);d.fill_diagonal_(float('inf'))
    if width_mode!='global':raise ValueError('This version requires recorded global width')
    widths=(d[torch.isfinite(d)&(d>1e-6)].median()/math.sqrt(2)).clamp_min(.1).repeat(candidates)
    return {'mean':mean,'projection':P,'scale':scale,'features':f,'centers':centers,'widths':widths,'rows':candidate_rows,'dictionary':gaussian(f,centers,widths),'target':token_error_target(base_logits,labels),'retained_variance_fraction':float(sv[:hidden_dimensions].square().sum()/sv.square().sum()),'width_mode':width_mode}

def make_lexical_adapter(state,*,hidden_width,vocabulary_size):
    basic={'mean','projection','feature_scale','centers','raw_widths','readout'};lexkeys={'lex_keys','lex_idf','lex_mean','lex_axes','lex_scale'}
    if set(state)!=basic|lexkeys:raise ValueError('Unexpected lexical checkpoint keys')
    mu,P,s,C,rho,W=[state[k]for k in ['mean','projection','feature_scale','centers','raw_widths','readout']]
    lex={k:state[k]for k in lexkeys};n=len(lex['lex_keys']);qdim=lex['lex_axes'].shape[1]
    if hidden_width!=768 or qdim!=256 or not 1<=n<=20000 or lex['lex_keys'].shape!=(n,2) or lex['lex_keys'].dtype!=torch.int64:raise ValueError('Invalid lexical vocabulary')
    if bool((lex['lex_keys'][:,0]<0).any()) or bool((lex['lex_keys']>=vocabulary_size).any()) or bool((lex['lex_keys'][:,1]<-1).any()):raise ValueError('Invalid lexical token ids')
    if len({tuple(k)for k in lex['lex_keys'].tolist()})!=n or lex['lex_idf'].shape!=(n,) or lex['lex_mean'].shape!=(n,) or lex['lex_scale'].shape!=(qdim,) or lex['lex_axes'].shape!=(n,qdim):raise ValueError('Invalid lexical projection')
    if mu.shape!=(hidden_width+qdim,) or P.ndim!=2 or P.shape[0]!=len(mu) or not 1<=P.shape[1]<=384:raise ValueError('Invalid joint projection')
    if C.ndim!=2 or not 1<=len(C)<=1024 or C.shape[1]!=P.shape[1] or s.shape!=(P.shape[1],) or rho.shape!=(len(C),) or W.shape!=(len(C),vocabulary_size):raise ValueError('Invalid Gaussian shapes')
    if any(not torch.isfinite(t).all()for t in state.values()) or not(s>0).all() or not(lex['lex_scale']>0).all() or not(lex['lex_idf']>0).all():raise ValueError('Invalid numerical state')
    a=LexicalGaussianAdapter(mu,P,s,C,torch.nn.functional.softplus(rho).clamp_min(1e-4),W,lex,hidden_width)
    projection=a.projection.detach().clone();del a._buffers['projection'];a.register_parameter('projection',torch.nn.Parameter(projection));a.load_state_dict(state,strict=True);a.clear_question();return a