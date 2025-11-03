from __future__ import print_function

import torch
import torch.nn as nn


class XASSupConLoss(nn.Module):
    """XAS Supervised Contrastive Learning.
    It also supports the unsupervised contrastive loss in SimCLR and supervised contrastive loss in SupCon"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, age_sigma = 10.2, age_weight = 0.1, sex_weight = 0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.age_sigma = age_sigma
        self.age_weight = age_weight
        self.sex_weight = sex_weight

    def forward(self, features, labels=None, ages=None, sex=None, mask=None):
        """
        Compute the contrastive loss.

        If both labels and mask are None, this reduces to the SimCLR unsupervised loss
        where each view of the same sample forms a positive pair.

        Args:
            features (Tensor): shape [batch_size, n_views, dim] or [batch_size, n_views, ...].
                If features have more than three dimensions, trailing dimensions are flattened.
            labels (Tensor, optional): shape [batch_size]. Class ids used to form positive pairs.
            ages (Tensor, optional): shape [batch_size] or [batch_size, 1]. Optional age term for weighting.
            sex (Tensor, optional): shape [batch_size] or [batch_size, 1]. Optional sex term for weighting.
            mask (Tensor, optional): shape [batch_size, batch_size]. Binary matrix where mask[i, j] = 1
                marks i and j as positives; may be asymmetric. When provided, this overrides labels.

        Returns:
            Tensor: scalar loss value.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # label similarity (same class -> 1)
            label_mask = torch.eq(labels, labels.T).float().to(device)

             # age similarity (closer ages -> higher similarity)
            if ages is not None and sex is not None:
                ages = ages.contiguous().view(-1, 1).float().to(device)
                age_diff = torch.abs(ages - ages.T)          # pairwise |age_i - age_j|

                # Gaussian (RBF) similarity
                age_sim = torch.exp(- (age_diff ** 2) / (2 * self.age_sigma ** 2))

                sex_mask = torch.eq(sex, sex.T).float().to(device) 

                pos_w = (1 + self.age_weight * age_sim) * (1 + self.sex_weight * sex_mask)
                mask  = label_mask * pos_w

            else:
                mask = label_mask
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
