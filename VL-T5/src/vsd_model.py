import torch
import torch.nn as nn


from modeling_t5 import VLT5
from modeling_bart import VLBart


class VLT5VSD(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, one_step_dec=False):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)
        # rel_labels = batch['target_relation_ids'].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            # rel_labels=rel_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, beam_with_prompt=False, one_step_dec=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result

class VLBartVSD(VLBart):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_head_and_prompt(self):
        self.model.encoder.prompt_embedding.requires_grad = True
        if self.config.deep_prompt:
            self.model.encoder.deep_prompt_embeddings.requires_grad = True

        for p in self.depth_conv.parameters():
            p.requires_grad = True
        
        for p in self.depth_linear.parameters():
            p.requires_grad = True
        
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        for p in self.bbox_encode.parameters():
            p.requires_grad = True

        for p in self.bbox_cls.parameters():
            p.requires_grad = True

    def train_step(self, batch, one_step_dec=False):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_depth = batch['vis_depth'].to(device)
        so_bbox = batch['raw_bbox'].to(device)
        task = batch["task"]

        lm_labels = batch["target_ids"].to(device)
        # rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # vis_depth = vis_depth,
            labels=lm_labels,
            task=task,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']
        # sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        # if rel_labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     predicate_loss = loss_fct(predicate_logits.view(-1, 9), rel_labels.view(-1))
        #     loss += predicate_loss
        
        result = {
            'loss': loss
        }
        return result

    
    def test_step(self, batch, beam_with_prompt=False, one_step_dec=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_depth = batch['vis_depth'].to(device)
        so_bbox = batch['raw_bbox'].to(device)
        task = batch["task"]

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # vis_depth = vis_depth,
            task=task,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result