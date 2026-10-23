"""Run in the user's torch environment; does not load target/draft checkpoints."""
import importlib.util
import os
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@unittest.skipIf(torch is None,'PyTorch not installed in this validation environment')
class HeadCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(os.environ.get('ACCEPTANCE_SOURCE_ROOT',Path(__file__).resolve().parents[2]))
        base=root/'speculators/src/speculators/models/dflash_prefix'
        cls.Head=load_file('tested_head',base/'acceptance_head.py').AcceptedLengthHead
        cls.loss=staticmethod(load_file('tested_loss',base/'acceptance_loss.py').acceptance_boundary_loss)
        cls.features=staticmethod(load_file('tested_features',base/'acceptance_features.py').build_acceptance_features)

    def test_only_head_changes(self):
        torch.manual_seed(1)
        draft=torch.nn.Linear(6,8).requires_grad_(False).eval()
        head=self.Head(hidden_size=8,num_proposals=3,projection_size=4,mlp_hidden_size=8)
        before={n:p.detach().clone() for n,p in draft.named_parameters()}
        head_before={n:p.detach().clone() for n,p in head.named_parameters()}
        with torch.no_grad():
            hidden=draft(torch.randn(5,3,6))
            ids=torch.tensor([[[1,2],[3,4],[5,6]]]).expand(5,-1,-1)
            unary=torch.randn(5,3,2)
            selected=ids[...,0]
            features=self.features(hidden,ids,unary,selected,unary,validate=True)
        opt=torch.optim.AdamW(head.parameters(),lr=.01)
        loss,_=self.loss(head(features),torch.tensor([0,1,2,3,1]),validate=True)
        loss.backward(); opt.step()
        self.assertFalse(features.requires_grad)
        self.assertTrue(all(p.grad is None and torch.equal(p,before[n]) for n,p in draft.named_parameters()))
        self.assertTrue(any(not torch.equal(p,head_before[n]) for n,p in head.named_parameters()))

    def test_asymmetric_cost(self):
        under=torch.tensor([[-5.,8.,-5.,-5.,-5.]])
        over=torch.tensor([[-5.,-5.,-5.,8.,-5.]])
        target=torch.tensor([2])
        a,_=self.loss(under,target,validate=True); b,_=self.loss(over,target,validate=True)
        self.assertGreater(a.item(),b.item())


if __name__=='__main__': unittest.main()
