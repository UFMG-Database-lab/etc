import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 3., reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, labels):
        assert logits.size(0) == len(labels), "Batch with diff sizes"
        assert logits.size(1) > labels.max(), "labels with wrong label"
        
        # src.index_put_(((torch.ones(index.size(0)).cumsum(0) - 1).long(), index), torch.ones_like(index))
        index1 = labels.long()
        index0 = (torch.ones_like(labels).cumsum(0) - 1).long()
        values = torch.ones_like(labels).float()
        
        target         = torch.zeros_like(logits).float()
        target         = target.index_put_((index0, index1), values)
        #target[labels] = 1.
        #print(logits.shape, labels.shape, target.shape)
        
        return sigmoid_focal_loss(logits, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)