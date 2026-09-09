import torch
import torch.nn as nn
import torch.nn.functional as F
import sonata
from info_nce import InfoNCE

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS, build_model

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=384, num_hidden_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
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

@MODELS.register_module("PA3FF")
class PA3FF(nn.Module):

    def __init__(self,
                 backbone_dim=None, 
                 output_dim=None,
                 use_hierarchy_losses=True, 
                 max_grouping_scale=2, 
                 freeze_backbone=True,
                 **kwargs):
        super().__init__()

        self.use_hierarchy_losses = use_hierarchy_losses
        self.max_grouping_scale = max_grouping_scale
        self.device = "cuda"
        self.quantile_transformer = None

        self.backbone = sonata.model.load(
            "./libs/sonata/ckpt/sonata.pth",
            custom_config=dict(
                enc_patch_size=[1024 for _ in range(5)],
                enable_flash=False,
            ),
        ).to(self.device)
        self.init_feat = None
        self.backbone_dim = backbone_dim

        self.instance_net = SimpleMLP(
            in_dim=backbone_dim + 6,
            out_dim=output_dim,
            hidden_dim=768,
            num_hidden_layers=6
        ).float()

        if freeze_backbone:
            for name, param in self.named_parameters():
                if 'instance_net' not in name:
                    param.requires_grad = False

        self.supcon = SupConLoss(temperature=0.07)
        self.infonce = InfoNCE(negative_mode='paired', temperature=0.07)

        self.testing = False

    def get_mlp(self, point_feat):
        instance_pass = self.instance_net(point_feat)

        epsilon = 1e-5
        norms = instance_pass.norm(dim=-1, keepdim=True)
        instance_pass = instance_pass / (norms + epsilon)

        return instance_pass

    def get_loss(self, input, pcd):
        point_feat = []
        with torch.no_grad():
            self.backbone.eval()
            for pcd_dict in pcd:
                point = self.backbone(pcd_dict)

                for _ in range(2):
                    assert "pooling_parent" in point.keys()
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    if self.testing: print("feat shape:", point.feat[inverse].shape, parent.feat.shape)
                    del point.feat
                    point = parent
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = point.feat[inverse]
                    point = parent
                point_feat.append(point.feat[point.inverse].detach())

            if self.testing: print("pcd shape:", input[0]["point"].shape, point_feat[0].shape)
        
        point_selected_feat = point_feat

        loss_dict = {}

        # Calculate GT labels for the positive and negative pairs

        input_ids = []
        instances = []

        torch.cuda.empty_cache()
        sample_size = 100
        for i in range(len(input)):
            input_id = input[i]["labels"]
            pcd_pos = torch.cat([torch.from_numpy(input[i]["point"]).cuda(), torch.from_numpy(input[i]["color"]).cuda()], dim=-1).float()
            raw_feat = point_selected_feat[i][..., :self.backbone_dim].float()
            instance = self.get_mlp(torch.cat([raw_feat, pcd_pos], dim=-1))

            labels = torch.unique(input_id)
            sampled_idxs = []
            for lb in labels:
                inds = (input_id == lb).nonzero(as_tuple=False).squeeze(1)
                num = inds.size(0)
                if num > sample_size:
                    perm = torch.randperm(num, device=inds.device)[:sample_size]
                    sel = inds[perm]
                else:
                    sel = inds
                sampled_idxs.append(sel)

            sampled_idxs = torch.cat(sampled_idxs, dim=0)
            instance_sampled = instance[sampled_idxs]
            input_id_sampled = input_id[sampled_idxs]

            instances.append(instance_sampled)
            input_ids.append(input_id_sampled)

        # for instance in instances: print("instance shape:", instance.shape)
        features = torch.cat(instances, dim=0).unsqueeze(1)
        # print(features.shape)
        features = F.normalize(features, p=2, dim=-1)
        labels = torch.cat(input_ids, dim=0)

        corr_loss = self.supcon(features, labels)

        emb = input[0]['embedding']
        P, _, D = features.shape
        # assert P == 2000 and D == 768
        L = emb.shape[0]

        pos = emb[labels]
        ids = torch.arange(L, device=labels.device).unsqueeze(0).expand(P, L)
        neg = emb.unsqueeze(0).expand(P, L, D)[ids != labels.unsqueeze(1)].view(P, L-1, D)

        features = features.squeeze().float()

        semantic_loss = self.infonce(features, pos, neg)

        loss_dict["corr_loss"] = corr_loss
        loss_dict["semantic_loss"] = semantic_loss
        loss_dict["instance_loss"] = corr_loss + semantic_loss

        return loss_dict
    
    def forward(self, input, testing_sonata=False):
        data = []
        for input_dict in input:          
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.cuda()
            data_dict = input_dict["pcd"]
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.cuda()
            data_dict["grid_size"] = 0.01
            data.append(data_dict)
        if self.training:
            loss_dict = self.get_loss(input, data)
            return loss_dict
        else:
            data_dict = data[0]
            with torch.no_grad():
                self.backbone.eval()
                point = self.backbone(data_dict)
                
                for _ in range(2):
                    assert "pooling_parent" in point.keys()
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    point = parent
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = point.feat[inverse]
                    point = parent
                point_feat = point.feat[point.inverse]
                # del self.backbone

            # point_feat = point_feat[input_dict["mapping"]]
            # print("point_feat shape:", point_feat.shape)
            if testing_sonata:
                return point_feat
            
            raw_feat = point_feat[..., :self.backbone_dim].float()
            pcd_pos = torch.cat([torch.from_numpy(input[0]["point"]).cuda(), torch.from_numpy(input[0]["color"]).cuda()], dim=-1).float()
            instance_feat = self.get_mlp(torch.cat([raw_feat, pcd_pos], dim=-1))
            
            return F.normalize(instance_feat, p=2, dim=-1), point_feat


