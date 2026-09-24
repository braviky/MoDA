
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("seq", "pos", "rot")


def _mlp(in_dim, hidden_dim, out_dim, bias=True):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=bias), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, bias=bias),
    )


class IdentityDirectedModalCoUpdate(nn.Module):
    
    def __init__(self, node_dim, pair_dim, hidden_dim=None, role_dim=3, semantic_edge_dim=8):
        super().__init__()
        h = int(hidden_dim or node_dim)
        self.node_dim, self.role_dim = node_dim, role_dim
        self.semantic_edge_dim = semantic_edge_dim
        self.identity = nn.Linear(role_dim, node_dim, bias=False)
        self.need = nn.Linear(node_dim, h, bias=False)
        self.supply = nn.Linear(node_dim, h, bias=False)
                                                                                    
        self.rel_target = nn.Linear(node_dim, h, bias=False)
        self.rel_source = nn.Linear(node_dim, h, bias=False)
        self.rel_pair = nn.Linear(pair_dim, h, bias=False)
        self.rel_geom = nn.Linear(13, h, bias=False)
        self.rel_semantic = nn.Linear(semantic_edge_dim, h, bias=False)
        self.rel_out = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.relation_mod = nn.Linear(h, h, bias=False)
        self.collect_q = nn.ModuleDict({m: nn.Linear(node_dim, h, bias=False) for m in MODALITIES})
        self.collect_k = nn.ModuleDict({m: nn.Linear(h, h, bias=False) for m in MODALITIES})
        self.collect_v = nn.ModuleDict({m: nn.Linear(h, node_dim, bias=False) for m in MODALITIES})
        self.collect_out = nn.ModuleDict({m: nn.Linear(node_dim, node_dim, bias=False) for m in MODALITIES})
        self.modality_embed = nn.Parameter(torch.randn(3, node_dim) * (node_dim ** -0.5))
        evidence_dim = 5 * node_dim
        self.evidence = nn.ModuleDict({m: _mlp(evidence_dim, h, node_dim) for m in MODALITIES})
        self.q = nn.Linear(2 * node_dim, h, bias=False)
        self.k = nn.Linear(3 * node_dim, h, bias=False)
        self.v = nn.Linear(3 * node_dim, node_dim, bias=False)
        self.transfer_out = nn.Linear(node_dim, node_dim, bias=False)
        self.norm = nn.ModuleDict({m: nn.LayerNorm(node_dim) for m in MODALITIES})

    def _sender_reliability(self, roles, rho):
                                                                          
                                                                                 
        generated = roles["generated"].to(rho.dtype)
        return 1.0 - generated + generated * rho[:, None]

    def forward(self, modality_states, hierarchy_contexts, role_probability, p_t, r_t,
                pair_feat, edge_features, roles, beta, receiver_masks):
        states = modality_states
        b, l, d = states["seq"].shape
        device, dtype = states["seq"].device, states["seq"].dtype
                                                                          
                                                                            
                                                                          
                                                                        
        valid_float = roles["valid"].to(dtype)
        centered = (role_probability - 1.0 / self.role_dim) * valid_float.unsqueeze(-1)
        z = self.identity(centered)
        rho = beta.to(dtype).clamp(0, 1)
        sender_rel = self._sender_reliability(roles, rho)
        valid = roles["valid"]
        edge_mask = valid[:, :, None] & valid[:, None, :] & ~torch.eye(l, device=device, dtype=torch.bool)[None]
        if edge_features is None:
            edge_features = pair_feat.new_zeros(b, l, l, self.semantic_edge_dim)
                                                                                
                                                                
        delta = p_t[:, None, :, :] - p_t[:, :, None, :]
        dist = delta.norm(dim=-1, keepdim=True)
        local_delta = torch.matmul(r_t[:, :, None].transpose(-1, -2), delta.unsqueeze(-1)).squeeze(-1)
        rel_rot = torch.matmul(r_t[:, :, None].transpose(-1, -2), r_t[:, None, :]).reshape(b, l, l, 9)
        geom = torch.cat([dist, local_delta, rel_rot], dim=-1)
        avg_state = torch.stack([states[m] for m in MODALITIES], dim=0).mean(dim=0)
        base = self.rel_target(avg_state)[:, :, None, :]
        base = base + self.rel_source(avg_state)[:, None, :, :]
        base = base + self.rel_pair(pair_feat) + self.rel_geom(geom) + self.rel_semantic(edge_features)
        base = self.rel_out(base)
        need = self.need(z).unsqueeze(2)
        supply = self.supply(z).unsqueeze(1)
                                                                                 
        carrier = base + self.relation_mod(base * torch.tanh(need) * torch.tanh(supply))
        carrier = carrier * sender_rel[:, None, :, None]
        collected = {}
        collect_attention = {}
        for mi, m in enumerate(MODALITIES):
            q = self.collect_q[m](states[m])
            k = self.collect_k[m](carrier)
            score = (q[:, :, None, :] * k).sum(-1) / math.sqrt(k.shape[-1])
            score = score.masked_fill(~edge_mask, torch.finfo(score.dtype).min)
            attn = F.softmax(score.float(), dim=-1).to(dtype) * edge_mask.to(dtype)
            denom = attn.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
            attn = attn / denom
            msg = (attn.unsqueeze(-1) * self.collect_v[m](carrier)).sum(dim=2)
            collected[m] = self.collect_out[m](msg)
            collect_attention[m] = attn
                                                                                 
        evid = {}
        for m in MODALITIES:
            idx = MODALITIES.index(m)
            evid[m] = self.evidence[m](torch.cat([
                states[m], collected[m], hierarchy_contexts[m], z,
                self.modality_embed[idx].view(1, 1, -1).expand(b, l, -1)
            ], dim=-1))
        updated = {}
        transfer_alpha = {}
        for ti, target_m in enumerate(MODALITIES):
            pieces = [evid[target_m]]
            for si, source_m in enumerate(MODALITIES):
                if source_m == target_m:
                    continue
                q = self.q(torch.cat([evid[target_m], self.modality_embed[ti].view(1,1,-1).expand(b,l,-1)], dim=-1))
                k = self.k(torch.cat([evid[source_m], self.modality_embed[si].view(1,1,-1).expand(b,l,-1), z], dim=-1))
                value = self.v(torch.cat([evid[source_m], self.modality_embed[si].view(1,1,-1).expand(b,l,-1), z], dim=-1))
                alpha = torch.tanh((q * k).sum(-1, keepdim=True) / math.sqrt(k.shape[-1]))
                                                                           
                                                                             
                                                                              
                pieces.append(alpha * self.transfer_out(value))
                transfer_alpha[source_m + "_to_" + target_m] = alpha
            update = self.norm[target_m](torch.stack(pieces, dim=0).sum(dim=0))
                                                                        
                                                                           
                                                                                
            update = rho[:, None, None] * update
            updated[target_m] = torch.where(
                receiver_masks[target_m][..., None], update, torch.zeros_like(update)
            )
        diagnostics = {
            "identity_centered_norm": centered.norm(dim=-1).mean().detach(),
            "sender_reliability": sender_rel.mean().detach(),
            "edge_density": edge_mask.to(dtype).mean().detach(),
            "modal_transfer_norm": torch.stack([updated[m].norm(dim=-1).mean() for m in MODALITIES]).mean().detach(),
        }
        return updated, {"carrier": carrier, "collected": collected, "evidence": evid,
                         "collect_attention": collect_attention, "transfer_alpha": transfer_alpha,
                         "updated": updated, "input_states": states,
                         "diagnostics": diagnostics, "receiver_masks": receiver_masks}



