import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mlsystem.config import ModelConfig
from mlsystem.model import LanguageModel
from mlsystem.genlora import attach,AdapterLinear
from mlsystem.ols_rbf import GaussianLogitAdapter,RBFAdaptedModel
from mlsystem.decoding import CachedDecoder

class DecodeTests(unittest.TestCase):
    def base(self,mode='standard'):
        torch.manual_seed(217)
        return LanguageModel(ModelConfig(vocab_size=512,context_length=64,d_model=64,num_layers=3,
            num_heads=4,num_kv_heads=2,d_ff=64,attention=mode)).eval()

    def check_chunks(self,model):
        tokens=torch.randint(0,512,(2,13));decoder=CachedDecoder(model)
        before={key:value.detach().clone() for key,value in model.state_dict().items()}
        end=0
        for width in [5,1,3,1,3]:
            next_end=end+width
            actual=decoder.last_logits(tokens[:,end:next_end])
            with torch.no_grad():expected=model(tokens[:,:next_end])[:,-1]
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6);end=next_end
        fresh=CachedDecoder(model)
        with torch.no_grad():expected=model(tokens[:,:2])[:,-1]
        torch.testing.assert_close(fresh.last_logits(tokens[:,:2]),expected,rtol=2e-5,atol=3e-6)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        with self.assertRaises(ValueError):decoder.last_logits(tokens[:1,:1])
        with self.assertRaises(ValueError):decoder.last_logits(torch.zeros(2,60,dtype=torch.long))

    def test_standard_and_hopfield_all_query_rows_match_full_forward(self):
        for mode in ['standard','hopfield']:self.check_chunks(self.base(mode))

    def test_nonzero_genlora_cache_matches_and_does_not_change_weights(self):
        model=self.base();attach(model,'rbf',4,groups=4,centers=5)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,AdapterLinear):module.gen_b.rbf.normal_(std=.005)
        self.check_chunks(model.eval())

    def test_ols_complete_vocabulary_residual_matches(self):
        model=self.base();adapter=GaussianLogitAdapter(torch.zeros(64),torch.eye(64)[:,:4],
            torch.ones(4),torch.randn(6,4),torch.ones(6),torch.randn(6,512)*.01)
        self.check_chunks(RBFAdaptedModel(model,adapter).eval())
