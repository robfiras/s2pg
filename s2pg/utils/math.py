import torch
import torch.nn.functional as F
from typing import Optional


class GailDiscriminatorLoss(torch.nn.modules.BCEWithLogitsLoss):

    def __init__(self, entcoeff=1e-3, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None) -> None:

        super(GailDiscriminatorLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)

        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.entcoeff = entcoeff

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # overrides original BCELoss
        bce_loss = torch.maximum(input, torch.zeros_like(input)) - input * target + torch.log \
            (1 + torch.exp(-torch.abs(input)))
        bce_loss = torch.mean(bce_loss)

        bernoulli_ent = self.entcoeff * torch.mean(self.logit_bernoulli_entropy(input))
        return bce_loss - bernoulli_ent

    def logit_bernoulli_entropy(self, logits):
        """
        Adapted from:
        https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
        """
        return (1. - self.sigmoid(logits)) * logits - self.logsigmoid(logits)


class LeastSquaresGailDiscriminatorLoss(torch.nn.Module):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_expert = target.bool()
        loss = torch.mean(torch.square(input[is_expert] - torch.ones_like(input[is_expert]))) + \
               torch.mean(torch.square(input[~is_expert] + torch.ones_like(input[~is_expert])))
        return loss
