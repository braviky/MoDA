import torch
import torch.nn as nn
import torch.nn.functional as F

from moda.utils.protein.constants import Fragment


def build_structured_edge_features(batch, mask_generate, mask_res, p_t=None):
    
    n, l = mask_generate.shape
    device = mask_generate.device
    valid = mask_res.bool()
    if batch is None:
        base = torch.zeros(n, l, l, 8, device=device)
        edge_mask = valid[:, :, None] & valid[:, None, :]
        return base, edge_mask

    fragment_type = batch.get("fragment_type", torch.zeros_like(mask_generate, dtype=torch.long)).to(device)
    cdr_flag = batch.get("cdr_flag", torch.zeros_like(fragment_type)).to(device)
    chain_nb = batch.get("chain_nb", torch.zeros_like(fragment_type)).to(device)
    anchor_flag = batch.get("anchor_flag", torch.zeros_like(mask_generate, dtype=torch.bool)).to(device)

    src_antigen = (fragment_type == int(Fragment.Antigen)) & valid
    src_ab = ((fragment_type == int(Fragment.Heavy)) | (fragment_type == int(Fragment.Light))) & valid
    src_gen = mask_generate.bool() & valid
    src_framework = src_ab & (~src_gen)
    src_anchor = anchor_flag.bool() & valid
    tgt_gen = mask_generate.bool() & valid

    same_chain = chain_nb[:, :, None] == chain_nb[:, None, :]
    same_cdr = (cdr_flag[:, :, None] == cdr_flag[:, None, :]) & (cdr_flag[:, :, None] > 0)
    edge_mask = valid[:, :, None] & valid[:, None, :]

    if p_t is None:
        proximity = torch.zeros(n, l, l, device=device)
    else:
        dist = torch.cdist(p_t, p_t)
        proximity = torch.exp(-dist).masked_fill(~edge_mask, 0.0)

    feats = torch.stack(
        [
            src_antigen[:, None, :].expand(-1, l, -1).float(),
            src_framework[:, None, :].expand(-1, l, -1).float(),
            src_gen[:, None, :].expand(-1, l, -1).float(),
            src_anchor[:, None, :].expand(-1, l, -1).float(),
            tgt_gen[:, :, None].expand(-1, -1, l).float(),
            same_chain.float(),
            same_cdr.float(),
            proximity.float(),
        ],
        dim=-1,
    )
    return feats * edge_mask.unsqueeze(-1).float(), edge_mask


class StructuredEvidenceCoordinator(nn.Module):
    

    def __init__(self, pair_feat_dim, num_heads, edge_feat_dim=8, hidden_dim=32, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.edge_feat_dim = int(edge_feat_dim)
        self.num_heads = int(num_heads)
        self.net = nn.Sequential(
            nn.Linear(pair_feat_dim + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pair_feat, edge_feat, edge_mask):
        if (not self.enabled) or edge_feat is None:
            return None
        value = self.net(torch.cat([pair_feat, edge_feat], dim=-1))
        value = value.masked_fill(~edge_mask.unsqueeze(-1), 0.0)
        return value


class NativeRiskHead(nn.Module):
    

    def __init__(self, node_feat_dim, hidden_dim=64, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, node_feat):
        if not self.enabled:
            return None
        return F.softplus(self.net(node_feat))

