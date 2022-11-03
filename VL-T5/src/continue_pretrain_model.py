
from modeling_t5 import VLT5

from vqa_model import VLT5VQA
from gqa_model import VLT5GQA
from nlvr_model import VLT5NLVR
from refcoco_model import VLT5RefCOCO
from caption_model import VLT5COCOCaption
from mmt_model import VLT5MMT
from vcr_model import VLT5VCR
from vsd_model import VLT5VSD
from classification_model import VLT5Classification

class TokenPermutation(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # def set_pretrain(self):
        

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        # vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        # vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            labels=lm_labels,
            reduce_loss=reduce_loss,
            task=task,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            task=task,
            **kwargs,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result

class ConstituencyParsing(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # def set_pretrain(self):
        

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        # vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        # vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            labels=lm_labels,
            reduce_loss=reduce_loss,
            task=task,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        input_ids = batch['input_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
            task=task,
            **kwargs,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result

class VLT5MultiTask(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'tp':
            return TokenPermutation.train_step(self, batch, **kwargs)
        elif task == 'ctp':
            return ConstituencyParsing.train_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.train_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.train_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.train_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.train_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.train_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.train_step(self, batch, **kwargs)
        elif task == 'vsd':
            return VLT5VSD.train_step(self, batch, **kwargs)

    def valid_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.valid_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.valid_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.valid_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.valid_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.valid_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.valid_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.valid_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.valid_step(self, batch, **kwargs)
        elif task == 'vsd':
            return VLT5VSD.valid_step(self, batch, **kwargs)

    def test_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.test_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.test_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.test_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.test_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.test_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.test_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.test_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.test_step(self, batch, **kwargs)
        elif task == 'vsd':
            return VLT5VSD.test_step(self, batch, **kwargs)
