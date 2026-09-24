import torch
import torch.nn as nn
import torch.nn.functional as F

from moda.utils.protein.constants import Fragment


def _masked_mean(x, mask):
    return (x * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp_min(1.0)


class SmoothNashSurplus(nn.Module):
    

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)

    def utilities(self, p_norm, batch, mask_generate, mask_res):
        n, l = mask_generate.shape
        device = p_norm.device
        if batch is None:
            return p_norm.new_zeros(n, 2)

        fragment_type = batch.get("fragment_type", torch.zeros_like(mask_generate, dtype=torch.long)).to(device)
        anchor_flag = batch.get("anchor_flag", torch.zeros_like(mask_generate, dtype=torch.bool)).to(device)
        valid = mask_res.bool()
        gen = mask_generate.bool() & valid
        antigen = (fragment_type == int(Fragment.Antigen)) & valid
        anchor = anchor_flag.bool() & valid

        epi = p_norm.new_zeros(n)
        fr = p_norm.new_zeros(n)
        for b in range(n):
            gen_pos = p_norm[b, gen[b]]
            if gen_pos.numel() == 0:
                continue
            ag_pos = p_norm[b, antigen[b]]
            if ag_pos.numel() > 0:
                epi[b] = torch.exp(-torch.cdist(gen_pos, ag_pos).min(dim=-1).values).mean()
            anchor_pos = p_norm[b, anchor[b]]
            if anchor_pos.numel() > 0:
                fr[b] = torch.exp(-torch.cdist(gen_pos, anchor_pos).min(dim=-1).values).mean()
        return torch.stack([epi, fr], dim=-1)

    def surplus(self, utility, disagreement=None, scale=None):
        if disagreement is None:
            disagreement = utility.detach().median(dim=0).values
        if scale is None:
            centered = utility.detach() - utility.detach().median(dim=0).values
            scale = centered.abs().median(dim=0).values.clamp_min(1e-3)
        z = (utility - disagreement[None, :]) / scale[None, :]
        return scale[None, :] * F.softplus(z)

    def loss(self, p_pred_norm, p_ref_norm, batch, mask_generate, mask_res):
        if not self.enabled or batch is None:
            return p_pred_norm.sum() * 0.0
        pred_u = self.utilities(p_pred_norm, batch, mask_generate, mask_res)
        ref_u = self.utilities(p_ref_norm.detach(), batch, mask_generate, mask_res)
        disagreement = ref_u.detach().median(dim=0).values
        centered = ref_u.detach() - disagreement[None, :]
        scale = centered.abs().median(dim=0).values.clamp_min(1e-3)
        pred_phi = torch.log(self.surplus(pred_u, disagreement, scale) + 1e-8).sum(dim=-1)
        ref_phi = torch.log(self.surplus(ref_u, disagreement, scale) + 1e-8).sum(dim=-1)
        return F.relu(ref_phi.detach() - pred_phi).mean()

