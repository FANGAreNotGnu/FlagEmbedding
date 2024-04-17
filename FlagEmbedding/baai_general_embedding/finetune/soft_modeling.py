from typing import Dict, Optional

import torch
from torch import nn, Tensor

from .modeling import BiEncoderModel, EncoderOutput


class SoftBiEncoderModel(BiEncoderModel):
    def __init__(
            self,
            model_name: str = None,
            normlized: bool = False,
            sentence_pooling_method: str = 'cls',
            negatives_cross_device: bool = False,
            temperature: float = 1.0,
            use_inbatch_neg: bool = True,
            soft_prune_ratio = 0.5,  # if it's 0.25, the lowest 25% loss will be truncated with a probability
            soft_prune_prob = 0.5,  # if it's 0.25, the lowest 25% loss will be truncated
        ):
        super().__init__(
            model_name=model_name,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            use_inbatch_neg=use_inbatch_neg,
        )
        self.soft_prune_ratio = soft_prune_ratio
        self.soft_prune_prob = soft_prune_prob
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)
            loss = self.prune_loss(loss)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def prune_loss(self, loss):
        assert loss.dim() == 1
        N = len(loss)
        mask = loss.clone().detach()
        thres = torch.quantile(mask, self.soft_prune_ratio)  # TODO
        mask = mask >= thres  # only large loss gets backproped

        prob_mask = torch.rand(mask.shape, device=mask.device) > self.soft_prune_prob  # if soft_prune_prob=0.25, 25% of the sample below threshold will be pruned (prob_mask = 0), else gets backproped
        mask = torch.logical_or(mask, prob_mask)
        
        loss = torch.sum(mask * loss) / (N * (1- self.soft_prune_ratio))
        return loss
