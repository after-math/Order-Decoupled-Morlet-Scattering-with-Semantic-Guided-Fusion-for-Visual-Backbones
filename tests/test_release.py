"""Main-text coverage and executable static/frozen training contracts."""
import sys,json,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from model import Experiment,classifier,load_imagenet
from tools.plan import jobs
from train import epoch_batches
from full_frame import letterbox
from static_original import StaticOriginalScattering
from order_scattering import OrderScattering
from e0kit.training import classification_sums

class ReleaseContracts(unittest.TestCase):
    def test_main_text_scope(self):
        self.assertEqual(len(list(jobs('main'))),98)
        self.assertEqual(len(list(jobs('limited'))),300)
        self.assertEqual(len(list(jobs('frozen'))),28)
        self.assertEqual(len(list(jobs('orders'))),42)
        self.assertEqual(len(list(jobs('components'))),20)
        self.assertEqual(len(list(jobs('gates'))),14)
        self.assertEqual({j['dataset'] for j in jobs('limited')},{'AID','Flowers17','Flowers102',*['KTHb_f'+str(i) for i in range(1,5)]})
        self.assertEqual({j['dataset'] for j in jobs('frozen')},{'AID','RSSCN7'})
        self.assertTrue(all(j['width']==128 for family in ['main','orders','components'] for j in jobs(family) if j['backbone']=='mobilevitv2_100'))

    def test_pretrained_metadata(self):
        metadata=json.loads((Path(__file__).resolve().parents[1]/'configs/imagenet.json').read_text())
        self.assertEqual(len(metadata),7)
        for row in metadata.values():
            self.assertEqual(len(row['mean']),3);self.assertEqual(len(row['std']),3)
            self.assertEqual(len(row['sha256']),64);self.assertTrue(row['url'].startswith('https://'))
        m=Experiment('mambaout_femto',7,frozen=True)
        self.assertIs(classifier(m.native),m.native.body.head.fc2)
        self.assertFalse(m.native.body.head.fc1.weight.requires_grad)
        self.assertTrue(m.native.body.head.fc2.weight.requires_grad)

    def test_imagenet_loader_preserves_task_head(self):
        for backbone in ['resnet18','efficientnet_b0','mambaout_femto']:
            with self.subTest(backbone=backbone),tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(1);source=Experiment(backbone,1000,'Native')
                torch.manual_seed(42);target=Experiment(backbone,7,'Native',frozen=True)
                initial=classifier(target.native).weight.detach().clone()
                path=Path(folder)/'weights.pth';torch.save({'model':source.native.body.state_dict()},path)
                load_imagenet(target,path,[.485,.456,.406],[.229,.224,.225])
                torch.testing.assert_close(classifier(target.native).weight,initial)
                excluded={n for n,p in target.native.body.named_parameters() if any(p is q for q in classifier(target.native).parameters())}
                for name,value in target.native.body.state_dict().items():
                    if name not in excluded:torch.testing.assert_close(value,source.native.body.state_dict()[name])

    def test_static_original_equivalence(self):
        model=Experiment('resnet18',7)
        q1,q2=model.branch.parameters_by_order();rgb=torch.rand(1,3,256,256)
        original=StaticOriginalScattering();standard=OrderScattering(checkpoint_chunks=False)
        with torch.no_grad():
            actual=original([rgb[0]],q1,q2)
            expected=standard(rgb,q1,q2)
            expected=tuple(F.adaptive_avg_pool2d(x.log1p(),(16,16)) for x in expected)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
        self.assertRaises(ValueError,original,[rgb[0]],q1,q2[None])

    def test_letterbox(self):
        mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
        image=torch.rand(3,64,96);canvas,boxes=letterbox([image],mean)
        self.assertEqual(boxes,[(42,0,171,256)])
        torch.testing.assert_close(canvas[0,:,:42],mean.reshape(3,1,1).expand(3,42,256))

    def test_real_static_backward(self):
        model=Experiment('resnet18',7,precision='fp32');model.train()
        loss=F.cross_entropy(model(torch.rand(1,3,256,256)),torch.tensor([1]));loss.backward()
        for p in [model.branch.bank.local,model.branch.second_bank.local,model.branch.gate.weight,model.alpha]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)

    def test_real_frozen_update(self):
        model=Experiment('resnet18',7,precision='fp32',frozen=True);model.train()
        old=model.native.body.bn1.running_mean.clone();head=classifier(model.native).weight.detach().clone()
        images=[torch.randint(0,256,(3,64,80),dtype=torch.uint8) for _ in range(2)]
        y=torch.tensor([1,2]);optimizer=torch.optim.AdamW(model.optimizer_groups());optimizer.zero_grad()
        with epoch_batches(model,dict(dataset='RSSCN7'),images,y,torch.arange(2),1,42,torch.device('cpu')) as stream:
            for rgb,target,partner,lam in stream:
                for i in range(len(target)):
                    loss=classification_sums(model(rgb[i:i+1]),target[i:i+1],None if partner is None else partner[i:i+1],lam)[0]/len(target)
                    loss.backward()
        optimizer.step()
        torch.testing.assert_close(model.native.body.bn1.running_mean,old)
        self.assertFalse(torch.equal(classifier(model.native).weight,head))
        self.assertIsNone(model.native.body.conv1.weight.grad)
        self.assertGreater(model.branch.second_bank.local.grad.abs().sum(),0)

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
