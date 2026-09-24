import torch
import torch.nn as nn
import torch.nn.functional as F

from moda.utils.protein.constants import BBHeavyAtom, Fragment, CDR


def _masked_mean(x, mask, dim, keepdim=False):
    weight = mask.float()
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim, keepdim=keepdim) / weight.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)


def _safe_ratio(num, den):
    return num.float() / den.float().clamp_min(1.0)


def _min_distance_summary(src_pos, src_mask, dst_pos, dst_mask):
    n = src_pos.size(0)
    out = src_pos.new_zeros(n, 2)
    for b in range(n):
        src = src_pos[b, src_mask[b]]
        dst = dst_pos[b, dst_mask[b]]
        if src.numel() == 0 or dst.numel() == 0:
            continue
        dist = torch.cdist(src, dst)
        nearest = dist.min(dim=-1).values
        out[b, 0] = nearest.mean()
        out[b, 1] = nearest.min()
    return out


def build_case_features(batch, mask_generate, mask_res):
    
    device = mask_generate.device
    n, l = mask_generate.shape
    valid = mask_res.bool()

    if batch is None:
        return torch.zeros(n, 12, device=device)

    fragment_type = batch.get("fragment_type", torch.zeros_like(mask_generate, dtype=torch.long)).to(device)
    cdr_flag = batch.get("cdr_flag", torch.zeros_like(fragment_type)).to(device)
    anchor_flag = batch.get("anchor_flag", torch.zeros_like(mask_generate, dtype=torch.bool)).to(device)

    antigen = valid & (fragment_type == int(Fragment.Antigen))
    antibody = valid & ((fragment_type == int(Fragment.Heavy)) | (fragment_type == int(Fragment.Light)))
    framework = antibody & (~mask_generate.bool())
    heavy_gen = mask_generate.bool() & (fragment_type == int(Fragment.Heavy))
    light_gen = mask_generate.bool() & (fragment_type == int(Fragment.Light))
    h3_gen = mask_generate.bool() & (cdr_flag == int(CDR.H3))
    l3_gen = mask_generate.bool() & (cdr_flag == int(CDR.L3))

    valid_count = valid.sum(dim=-1)
    gen_count = mask_generate.bool().sum(dim=-1)
    antigen_count = antigen.sum(dim=-1)
    framework_count = framework.sum(dim=-1)
    anchor_count = anchor_flag.bool().sum(dim=-1)

    scalar = [
        _safe_ratio(gen_count, valid_count),
        _safe_ratio(anchor_count, valid_count),
        _safe_ratio(antigen_count, valid_count),
        _safe_ratio(framework_count, valid_count),
        _safe_ratio(heavy_gen.sum(dim=-1), gen_count),
        _safe_ratio(light_gen.sum(dim=-1), gen_count),
        _safe_ratio(h3_gen.sum(dim=-1), gen_count),
        _safe_ratio(l3_gen.sum(dim=-1), gen_count),
    ]

    geom = torch.zeros(n, 4, device=device)
    if "pos_heavyatom" in batch and "mask_heavyatom" in batch:
        pos_ca = batch["pos_heavyatom"][:, :, BBHeavyAtom.CA].to(device)
        atom_mask = batch["mask_heavyatom"][:, :, BBHeavyAtom.CA].to(device).bool()
        obs_anchor = anchor_flag.bool() & atom_mask
        obs_antigen = antigen & atom_mask
        obs_framework = framework & atom_mask
        ag_anchor = _min_distance_summary(pos_ca, obs_anchor, pos_ca, obs_antigen)
        fr_anchor = _min_distance_summary(pos_ca, obs_anchor, pos_ca, obs_framework)
        geom = torch.cat([ag_anchor, fr_anchor], dim=-1) / 10.0

    return torch.cat([torch.stack(scalar, dim=-1), geom], dim=-1)


class CaseAdaptiveClock(nn.Module):
    

    def __init__(self, case_dim=12, hidden_dim=32, num_modalities=3, quadrature_nodes=16, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.num_modalities = int(num_modalities)
        self.quadrature_nodes = int(quadrature_nodes)
        self.net = nn.Sequential(
            nn.Linear(case_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modalities),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _activity(self, lam, case_features):
        if lam.dim() == 1:
            lam = lam[:, None]
        time_feat = torch.cat([lam, torch.sin(lam), torch.cos(lam)], dim=-1)
        x = torch.cat([case_features, time_feat], dim=-1)
        return F.softplus(self.net(x)) + 1e-6

    def forward(self, lam, case_features):
        if (not self.enabled) or case_features is None:
            base = lam.clamp(0.0, 1.0)
            return {"pos": base, "rot": base, "seq": base, "activity": None}

        n = lam.size(0)
        grid = torch.linspace(0.0, 1.0, self.quadrature_nodes + 1, device=lam.device, dtype=lam.dtype)
        grid = grid.unsqueeze(0).expand(n, -1)
        case_grid = case_features.unsqueeze(1).expand(-1, grid.size(1), -1)
        activity = self._activity(grid.reshape(-1), case_grid.reshape(-1, case_features.size(-1)))
        activity = activity.view(n, grid.size(1), self.num_modalities)

        delta = grid[:, 1:] - grid[:, :-1]
        area = 0.5 * (activity[:, 1:] + activity[:, :-1]) * delta[:, :, None]
        cumulative = torch.cat([activity.new_zeros(n, 1, self.num_modalities), area.cumsum(dim=1)], dim=1)
        denom = cumulative[:, -1].clamp_min(1e-6)
        tau_grid = cumulative / denom[:, None, :]

        lam_clamped = lam.clamp(0.0, 1.0)
        pos = torch.bucketize(lam_clamped.detach(), grid[0], right=True).clamp(1, grid.size(1) - 1)
        left = pos - 1
        right = pos
        left_tau = tau_grid.gather(1, left[:, None, None].expand(-1, 1, self.num_modalities)).squeeze(1)
        right_tau = tau_grid.gather(1, right[:, None, None].expand(-1, 1, self.num_modalities)).squeeze(1)
        left_lam = grid.gather(1, left[:, None]).squeeze(1)
        right_lam = grid.gather(1, right[:, None]).squeeze(1)
        frac = ((lam_clamped - left_lam) / (right_lam - left_lam).clamp_min(1e-6))[:, None]
        tau = left_tau + frac * (right_tau - left_tau)
        tau = tau.clamp(0.0, 1.0)
        return {"pos": tau[:, 0], "rot": tau[:, 1], "seq": tau[:, 2], "activity": activity}

