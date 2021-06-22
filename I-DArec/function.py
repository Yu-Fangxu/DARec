from torch.autograd import Function
import torch.nn as nn
import torch


class MRMSELoss(nn.Module):
    """
    MaskedRootMSELoss() uses only observed ratings.
    According to docs(https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html),
    'mean' is set by default for 'reduction' and can be avoided by 'reduction="sum"'
    """

    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)

    def forward(self, pred, rating):
        mask = rating != 0
        masked_pred = pred * mask.float()
        num_observed = torch.sum(mask).cuda() if self.reduction == 'mean' else torch.Tensor([1.]).cuda()
        # loss = torch.sqrt(self.mse(masked_pred, rating) / num_observed)
        loss = self.mse(masked_pred, rating) / num_observed
        return loss, mask

class DArec_Loss(nn.Module):
    def __init__(self, reduction='sum', lamda=0.001, u=1.0, beta=0.001):
        super(DArec_Loss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)
        self.rmse = MRMSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.u = u
        self.beta = beta
    def forward(self, class_output, source_prediction, target_prediction, source_rating, target_rating, labels):
        """
        :param class_output: Domain Classifier分类结果
        :param source_prediction: Rating Predictor在Source Data上预测结果
        :param target_prediction: Rating Predictor在Target Data上预测结果
        :param source_rating: Source Data的GT
        :param target_rating: Target Data的GT
        :param labels: data class
        :return: Loss
        """
        source_loss, source_mask = self.rmse(source_prediction, source_rating)
        target_loss, target_mask = self.rmse(target_prediction, target_rating)
        loss_pred = source_loss + self.beta * target_loss
        loss_dom = self.u * self.cross_entropy(class_output, labels)

        return loss_pred + loss_dom, source_mask, target_mask

