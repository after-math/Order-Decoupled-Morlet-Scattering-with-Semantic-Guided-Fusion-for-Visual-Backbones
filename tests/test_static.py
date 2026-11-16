"""Contracts for real filter banks, seven backbones, freezing, and donor mappings."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from model import Experiment,BACKBONES,classifier
from interventions import donor_map,context_cache
from data import train_indices
from e0kit.scattering import MorletKernels
from order_scattering import OrderScattering,ReferenceOrderScattering

torch.set_num_threads(1)

class CheapScattering(nn.Module):
    def forward(self,rgb,*parameters):
        # Only the backbone adapter test uses a stand-in; real FFT is tested below.
        return torch.ones(len(rgb),54,32,32),torch.ones(len(rgb),324,32,32)

class Contracts(unittest.TestCase):
    def test_seven_backbones(self):
        for name in BACKBONES:
            with self.subTest(name=name):
                torch.manual_seed(42)
                m=Experiment(name,7,width=128,precision='fp32').eval()
                m.branch.scattering=CheapScattering()
                with torch.no_grad():y=m(torch.rand(1,3,256,256))
                self.assertEqual(tuple(y.shape),(1,7));self.assertTrue(torch.isfinite(y).all())
                groups=m.optimizer_groups();ids=[id(p) for g in groups for p in g['params']]
                self.assertEqual(len(ids),len(set(ids)))
                self.assertEqual(set(ids),{id(p) for p in m.parameters() if p.requires_grad})

    def test_static_bank_contracts(self):
        for variant in ['Shared','FixedBanks','FirstOrderOnly','SecondOrderOnly','NoGate','FixedAlpha','FullStatic']:
            m=Experiment('resnet18',7,variant,width=128)
            a,b=m.branch.parameters_by_order()
            self.assertEqual(a.shape,(18,4));self.assertEqual(b.shape,(18,4))
            self.assertFalse(any('generator' in n or 'offset' in n for n,_ in m.named_parameters()))
            if variant=='Shared':
                torch.testing.assert_close(b,a[m.branch.shared_indices])
                b.sum().backward();self.assertGreater(m.branch.bank.local.grad.abs().sum(),0)
            if variant=='FixedBanks':self.assertFalse(m.branch.bank.local.requires_grad or m.branch.second_bank.local.requires_grad)
            if variant=='NoGate':self.assertFalse(m.branch.gate.weight.requires_grad)
            if variant=='FixedAlpha':self.assertFalse(m.alpha.requires_grad)
            if variant=='FullStatic':self.assertIsNot(m.branch.bank.local,m.branch.second_bank.local)
            psi=MorletKernels()(a.detach())
            torch.testing.assert_close(psi.mean((-2,-1)),torch.zeros(18,dtype=psi.dtype),atol=1e-7,rtol=0)
            torch.testing.assert_close(psi.abs().square().sum((-2,-1)),torch.ones(18))

    def test_own_context_identity(self):
        m=Experiment('resnet18',7,precision='fp32').eval()
        m.branch.scattering=CheapScattering()
        images=torch.randint(0,256,(2,3,256,256),dtype=torch.uint8)
        with torch.no_grad():
            contexts=context_cache(m,images,torch.device('cpu'),2)
            own=m(images.float()/255)
            supplied=m(images.float()/255,gate_context=contexts)
        torch.testing.assert_close(own,supplied,atol=1e-6,rtol=1e-5)

    def test_freeze(self):
        for name in BACKBONES:
            m=Experiment(name,7,frozen=True);m.train()
            self.assertFalse(m.native.training)
            head_ids={id(p) for p in classifier(m.native).parameters()}
            self.assertEqual({id(p) for p in m.native.parameters() if p.requires_grad},head_ids)
            for module in m.native.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):self.assertFalse(module.training)

    def test_donors(self):
        labels=np.repeat(np.arange(7),60)
        for mode in ['same','other','random']:
            for seed in [42,43,44]:
                a=donor_map(labels,mode,seed);b=donor_map(labels,mode,seed)
                np.testing.assert_array_equal(a,b)
                self.assertEqual(len(set(a)),len(labels));self.assertTrue(np.all(a!=np.arange(len(a))))

    def test_nested_labels(self):
        rows=[dict(relative_path=f'{label}/{i}.jpg',label=label) for label in range(3) for i in range(20)]
        m=dict(dataset='test',classes=['a','b','c'],splits=dict(train=rows))
        a=train_indices(m,5);b=train_indices(m,10)
        self.assertTrue(set(a)<set(b));self.assertEqual(len(a),15)

    def test_real_fft_equivalence(self):
        torch.manual_seed(42)
        model=Experiment('resnet18',7)
        first,second=model.branch.parameters_by_order()
        fast=OrderScattering(checkpoint_chunks=False);fast.single_image_reference=False
        reference=ReferenceOrderScattering(checkpoint_chunks=False)
        rgb=torch.rand(1,3,256,256)
        with torch.no_grad():
            actual=fast(rgb,first,second);expected=reference(rgb,first,second)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
