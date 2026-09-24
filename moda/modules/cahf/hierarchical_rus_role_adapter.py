

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EntityLoRA(nn.Module):
    

    def __init__(self, dim, num_entities, rank):
        super().__init__()
        self.rank = int(rank)
        self.a = nn.Parameter(torch.empty(num_entities, self.rank, dim))
        self.b = nn.Parameter(torch.empty(num_entities, dim, self.rank))
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)

    def forward(self, x, entity_index):
        a = self.a[entity_index]
        b = self.b[entity_index]
        low = torch.matmul(x, a.transpose(0, 1))
        return torch.matmul(low, b.transpose(0, 1)) / (self.rank ** 0.5)


class _MaskedEntityEncoder(nn.Module):
    

    def __init__(self, dim, num_entities, rank, num_heads=4):
        super().__init__()
        self.pre_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.entity_lora = _EntityLoRA(dim, num_entities, rank)
        self.pre_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x, mask, entity_index):
        mask = mask.bool()
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        h = self.pre_attn(x) * mask.unsqueeze(-1).to(x.dtype)
        attn, _ = self.attn(h, h, h, key_padding_mask=~safe_mask, need_weights=False)
        x = x + attn + self.entity_lora(h, entity_index)
        x = x + self.ffn(self.pre_ffn(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class _EntitySetEncoder(nn.Module):
    

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = int(num_heads)
        self.pre_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.pre_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, tokens, mask, confidence=None):
        
        mask = mask.bool()
        h = self.pre_attn(tokens) * mask.unsqueeze(-1).to(tokens.dtype)
        attn_mask = None
        if confidence is None:
            key_mask = mask
            has_evidence = mask.any(dim=1, keepdim=True)
        else:
            confidence = confidence.to(dtype=tokens.dtype).clamp(0.0, 1.0)
            key_mask = mask & (confidence > 0)
            has_evidence = key_mask.any(dim=1, keepdim=True)
                                                                              
                                                                           
                                                                         
            log_conf = confidence.float().clamp_min(torch.finfo(torch.float32).tiny).log()
            log_conf = log_conf[:, None, :].expand(-1, tokens.size(1), -1)
            attn_mask = log_conf[:, None].expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask.reshape(-1, tokens.size(1), tokens.size(1))
            attn_mask = attn_mask.to(dtype=tokens.dtype)

        safe_mask = key_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            if attn_mask is not None:
                attn_mask = attn_mask.clone()
                rows = empty[:, None].expand(-1, self.num_heads).reshape(-1)
                attn_mask[rows] = 0
        key_padding_mask = ~safe_mask
        if attn_mask is not None:
                                                                             
                                                                         
            additive_key_mask = torch.zeros_like(confidence, dtype=tokens.dtype)
            additive_key_mask = additive_key_mask.masked_fill(
                key_padding_mask, float("-inf")
            )
            key_padding_mask = additive_key_mask
        attended, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            need_weights=False,
        )
        attended = attended * has_evidence.unsqueeze(-1).to(attended.dtype)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.pre_ffn(tokens))
        return tokens * mask.unsqueeze(-1).to(tokens.dtype)


class _ConstraintReader(nn.Module):
    

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.unique_value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(dim * 3 + 1), nn.Linear(dim * 3 + 1, dim),
            nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, query, tokens, unique, key_mask, query_mask, observability):
        key_mask = key_mask.bool()
        query_mask = query_mask.bool()
        logits = torch.matmul(self.query(query), self.key(tokens).transpose(1, 2))
        logits = logits / (query.size(-1) ** 0.5)
        pair_mask = query_mask[:, :, None] & key_mask[:, None, :]
        weights = HierarchicalRUSRoleAdapter._masked_softmax(logits, pair_mask, dim=-1)
        attended = torch.bmm(weights, self.value(tokens))
        u = self.unique_value(unique)[:, None, :].expand_as(query)
        q_e = observability.to(query.dtype)[:, None, None].expand(-1, query.size(1), 1)
        message = self.out(torch.cat([query, attended, u, q_e], dim=-1))
        return message * query_mask.unsqueeze(-1).to(message.dtype)


class HierarchicalRUSRoleAdapter(nn.Module):
    

                                                                             
                                                                            
    LEVELS = ("complex", "pair_hl", "pair_ab_ag", "chain", "region")
    ENTITY_LEVELS = (
        "complex", "pair_hl", "pair_ab_ag", "chain_h", "chain_l",
        "region_h1", "region_h2", "region_h3",
        "region_l1", "region_l2", "region_l3",
    )
    PAIR_TYPES = ("hl", "ab_ag")

    def __init__(self, node_dim, pair_dim, enabled=True, num_roles=3, neighbor_topk=None):
        super().__init__()
        self.enabled = bool(enabled)
        self.node_dim = int(node_dim)
        self.pair_dim = int(pair_dim)
        self.num_roles = int(num_roles)
        if self.num_roles < 2:
            raise ValueError("num_roles must be at least 2 for centered role amplitudes")
                                                                               
                                                                          
        self.neighbor_topk = neighbor_topk
        hidden = max(32, node_dim // 2)

        self.time_embed = nn.Sequential(
            nn.Linear(3, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.pair_bias = nn.Linear(pair_dim, node_dim)
                                                                             
                                                                            
                                                                       
                                                                               
                                                                          
                                                                             
                                                                    
        lora_rank = max(1, node_dim // 16)
        self.chain_encoder = _MaskedEntityEncoder(node_dim, 2, lora_rank)
        self.region_encoder = _MaskedEntityEncoder(node_dim, 6, lora_rank)
        self.complex_encoder = _EntitySetEncoder(node_dim)
        self.entity_rus_encoder = _EntitySetEncoder(node_dim)
                                                                           
                                                                           
                                                               
        self.pair_hl_encoder = nn.Sequential(
            nn.LayerNorm(node_dim * 4), nn.Linear(node_dim * 4, node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.pair_hl_query = nn.Linear(node_dim, node_dim, bias=False)
        self.pair_hl_key = nn.Linear(node_dim, node_dim, bias=False)
        self.pair_hl_value = nn.Linear(node_dim, node_dim, bias=False)
                                                                            
                                                                              
                                                                                
        self.pair_abag_ab_encoder = nn.Sequential(
            nn.LayerNorm(node_dim * 4), nn.Linear(node_dim * 4, node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.pair_abag_ag_encoder = nn.Sequential(
            nn.LayerNorm(node_dim * 4), nn.Linear(node_dim * 4, node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
                                                                          
                                                                            
                                                     
        self.spatial_relation_score = nn.Sequential(
            nn.LayerNorm(node_dim * 2 + 4),
            nn.Linear(node_dim * 2 + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

                                                                         
                                                                           
        self.rus_shared = nn.Sequential(
            nn.Linear(node_dim * len(self.ENTITY_LEVELS) + node_dim + len(self.ENTITY_LEVELS), hidden),
            nn.SiLU(), nn.Linear(hidden, node_dim)
        )
        self.rus_unique = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(node_dim), nn.Linear(node_dim, hidden), nn.SiLU(), nn.Linear(hidden, node_dim))
             for _ in self.ENTITY_LEVELS]
        )
        self.rus_synergy = nn.Sequential(
            nn.Linear(node_dim * len(self.ENTITY_LEVELS) + node_dim + len(self.ENTITY_LEVELS), hidden),
            nn.SiLU(), nn.Linear(hidden, node_dim)
        )

                                                                              
        self.constraint_readers = nn.ModuleList([
            _ConstraintReader(node_dim) for _ in self.ENTITY_LEVELS
        ])
                                                                           
                                                                              
                                                                           
                                                                                
        self.num_property_classes = 6
        property_index = torch.tensor([
            3, 5, 0, 0, 4, 5, 1, 3, 1, 3,
            3, 2, 5, 2, 1, 2, 2, 3, 4, 4,
        ], dtype=torch.long)
        self.register_buffer("residue_property_index", property_index, persistent=False)
        self.constraint_sequence_heads = nn.ModuleList([
            nn.Linear(node_dim, self.num_property_classes) for _ in self.LEVELS
        ])
        self.constraint_geometry_heads = nn.ModuleList([
            nn.Linear(node_dim, 3) for _ in self.LEVELS
        ])
                                                                            
                                                                               
        self.identity_self_sequence = nn.Linear(node_dim, self.num_property_classes)
        self.identity_self_geometry = nn.Linear(node_dim, 6)
                                                                             
                                                                                  
        self.identity_game_sequence = nn.Linear(node_dim, self.num_property_classes * 2)
        self.identity_game_geometry = nn.Linear(node_dim, 9)
        self.rus_vicreg = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(node_dim), nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
            for _ in self.ENTITY_LEVELS
        ])
                                                                         
                                                               
        self.self_query = nn.Sequential(
            nn.LayerNorm(node_dim * (len(self.LEVELS) + 3)), nn.Linear(node_dim * (len(self.LEVELS) + 3), node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.role_prototypes = nn.Parameter(torch.randn(self.num_roles, node_dim))
        self.shared_role_mod = nn.Linear(node_dim, self.num_roles)
        self.unique_role_mod = nn.Linear(node_dim * len(self.LEVELS), self.num_roles)
                                                                               
                                                                           
        self.game_context = nn.Sequential(
            nn.LayerNorm(node_dim * (len(self.LEVELS) + 3)),
            nn.Linear(node_dim * (len(self.LEVELS) + 3), node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.synergy_edge = nn.Sequential(
            nn.Linear(node_dim * 2 + node_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
                                                                           
                                                                           
                                                                              
        self.pair_neighbor_score = nn.Linear(pair_dim, 1, bias=False)
                                                                           
                                                                            
                                                             
        self.payoff_head = nn.Sequential(
            nn.Linear(node_dim * 4, hidden), nn.SiLU(),
            nn.Linear(hidden, self.num_roles * self.num_roles)
        )
        self.role_value = nn.Linear(self.num_roles, node_dim, bias=False)

                                                                             
                                                                            
        self.semantic_clock_log_gamma = nn.Parameter(torch.zeros(()))

                                                                            
                                                                           
                                                                          
                                                                      
        self.last = None

    @staticmethod
    def _masked_global(x, mask):
        w = mask.to(x.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * w.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _safe_unit(x):
        
                                                                          
                                                                             
                                                                         
        work = x.float()
        eps = torch.finfo(work.dtype).eps
        denom = work.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()
        return (work / denom).to(dtype=x.dtype)

    @staticmethod
    def _case_anchor_role_logits(logits, anchor_mask):
        
        centered = logits - logits.mean(dim=-1, keepdim=True)
        weight = anchor_mask.to(torch.float32).unsqueeze(-1)
        centered_f = centered.float()
        count = (weight.sum(dim=1, keepdim=True) * logits.size(-1)).clamp_min(1.0)
        mean_square = (centered_f.square() * weight).sum(dim=(1, 2), keepdim=True) / count
        rms_floor = 1e-12
        rms = mean_square.clamp_min(rms_floor).sqrt()
        return (centered_f / rms).to(logits.dtype)

    @staticmethod
    def _masked_softmax(logits, mask, dim=-1):
        
        mask = mask.bool()
        has_valid = mask.any(dim=dim, keepdim=True)
        min_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, min_value)
        safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
        weights = F.softmax(safe_logits.float(), dim=dim).to(logits.dtype)
        weights = weights * mask.to(weights.dtype)
        denom = weights.sum(dim=dim, keepdim=True)
        safe_denom = torch.where(has_valid, denom, torch.ones_like(denom))
        weights = weights / safe_denom
        return torch.where(has_valid, weights, torch.zeros_like(weights))

    @staticmethod
    def _group_context(x, group, valid, positive_only=False):
        same = (group[:, :, None] == group[:, None, :])
        if positive_only:
            same = same & (group[:, :, None] >= 0) & (group[:, None, :] >= 0)
        same = same & valid[:, :, None] & valid[:, None, :]
        weights = same.to(x.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bij,bjd->bid", weights, x) / denom

    @staticmethod
    def _relation_context(x, source_mask, query_mask, valid):
        
        rel = query_mask[:, :, None] & source_mask[:, None, :]
        rel = rel & valid[:, :, None] & valid[:, None, :]
        weights = rel.to(x.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        context = torch.einsum("bij,bjd->bid", weights, x) / denom
        return context, rel.any(dim=-1)

    def _cross_relation_context(self, x, source_mask, query_mask, valid):
        
        rel = query_mask[:, :, None] & source_mask[:, None, :]
        rel = rel & valid[:, :, None] & valid[:, None, :]
        q = self.pair_hl_query(x)
        k = self.pair_hl_key(x)
        v = self.pair_hl_value(x)
        logits = torch.matmul(q, k.transpose(1, 2)) / (self.node_dim ** 0.5)
        weights = self._masked_softmax(logits, rel, dim=-1)
        return torch.matmul(weights, v), rel.any(dim=-1)

    def _spatial_relation_context(self, x, positions, source_mask, query_mask, valid):
        
        rel = query_mask[:, :, None] & source_mask[:, None, :]
        rel = rel & valid[:, :, None] & valid[:, None, :]
        delta = positions[:, None, :, :] - positions[:, :, None, :]
        delta_norm = delta.norm(dim=-1, keepdim=True)
        query = x[:, :, None, :].expand(-1, -1, x.size(1), -1)
        source = x[:, None, :, :].expand(-1, x.size(1), -1, -1)
        score_input = torch.cat([query, source, delta, delta_norm], dim=-1)
        logits = self.spatial_relation_score(score_input).squeeze(-1)
        weights = self._masked_softmax(logits, rel, dim=-1)
        context = torch.einsum("bij,bjd->bid", weights, x)
        return context, rel.any(dim=-1)

    def _metadata(self, base, batch, valid):
        b, length, _ = base.shape
        device = base.device
        default_chain = torch.arange(length, device=device)[None].expand(b, -1)
        zeros = torch.zeros((b, length), dtype=torch.long, device=device)

        if batch is None:
            chain_nb = default_chain
            fragment = zeros
            cdr = zeros
        else:
            def get(name, default):
                value = batch.get(name, default)
                return value.to(device) if torch.is_tensor(value) else default
            chain_nb = get("chain_nb", default_chain)
            fragment = get("fragment_type", zeros)
            cdr = get("cdr_flag", zeros)

                                                                          
                                                                              
        region = torch.where((cdr > 0) & (cdr <= 6), cdr - 1, torch.full_like(cdr, -1))
        heavy = (fragment == 1) & valid
        light = (fragment == 2) & valid
        antibody = heavy | light
        chain_type = torch.where(heavy, torch.zeros_like(fragment),
                                 torch.where(light, torch.ones_like(fragment), -1))
        antigen = valid & ~antibody
        return {
            "chain_nb": chain_nb,
            "fragment": fragment,
            "cdr": cdr,
            "region": region,
            "chain_type": chain_type,
            "heavy": heavy,
            "light": light,
            "antibody": antibody,
            "antigen": antigen,
        }

    def _routed_levels(
        self, base, pair_feat, positions, batch, valid, generated, case_reliability
    ):
        meta = self._metadata(base, batch, valid)

                                                                                
                                                                               
                                                                        
        chain_maps = []
        for entity_index, entity_mask in enumerate((meta["heavy"], meta["light"])):
            chain_maps.append(self.chain_encoder(base, entity_mask, entity_index))
        region_maps = []
        for entity_index in range(6):
            region_mask = (meta["region"] == entity_index) & valid
            region_maps.append(self.region_encoder(base, region_mask, entity_index))

                                                                             
                                             
        hl_h, hl_h_valid = self._cross_relation_context(
            base, meta["light"], meta["heavy"], valid
        )
        hl_l, hl_l_valid = self._cross_relation_context(
            base, meta["heavy"], meta["light"], valid
        )
        hl_ctx = torch.where(
            meta["heavy"].unsqueeze(-1), hl_h,
            torch.where(meta["light"].unsqueeze(-1), hl_l, torch.zeros_like(hl_h))
        )
        hl_valid = torch.where(
            meta["heavy"], hl_h_valid,
            torch.where(meta["light"], hl_l_valid, torch.zeros_like(hl_h_valid))
        )
        hl_rel = torch.cat([base, hl_ctx, base * hl_ctx, (base - hl_ctx).abs()], dim=-1)
        hl_ctx = self.pair_hl_encoder(hl_rel) * hl_valid.unsqueeze(-1).to(base.dtype)

                                                                       
        abag_ab, abag_ab_valid = self._spatial_relation_context(
            base, positions, meta["antigen"], meta["antibody"], valid
        )
        abag_ag_h, abag_ag_h_valid = self._spatial_relation_context(
            base, positions, meta["heavy"], meta["antigen"], valid
        )
        abag_ag_l, abag_ag_l_valid = self._spatial_relation_context(
            base, positions, meta["light"], meta["antigen"], valid
        )
        abag_ag_valid = abag_ag_h_valid | abag_ag_l_valid
        opposite_chain = torch.where(
            meta["heavy"].unsqueeze(-1), hl_h,
            torch.where(meta["light"].unsqueeze(-1), hl_l, torch.zeros_like(hl_h))
        )
        abag_ab_input = torch.cat([base, abag_ab, opposite_chain, base * abag_ab], dim=-1)
        abag_ag_input = torch.cat([
            base, abag_ag_h, abag_ag_l, base * (abag_ag_h + abag_ag_l)
        ], dim=-1)
        abag_ab_ctx = self.pair_abag_ab_encoder(abag_ab_input)
        abag_ag_ctx = self.pair_abag_ag_encoder(abag_ag_input)
        abag_ctx = torch.where(
            meta["antibody"].unsqueeze(-1), abag_ab_ctx,
            torch.where(meta["antigen"].unsqueeze(-1), abag_ag_ctx, torch.zeros_like(abag_ab_ctx))
        )
        abag_valid = torch.where(
            meta["antibody"], abag_ab_valid,
            torch.where(meta["antigen"], abag_ag_valid, torch.zeros_like(abag_ab_valid))
        )
        abag_ctx = abag_ctx * abag_valid.unsqueeze(-1).to(base.dtype)

                                                                            
        global_token = self._masked_global(base, valid)
        heavy_token = self._masked_global(base, meta["heavy"])
        light_token = self._masked_global(base, meta["light"])
        antigen_token = self._masked_global(base, meta["antigen"])
        interface_mask = meta["antibody"] & abag_valid
        interface_token = self._masked_global(abag_ctx, interface_mask)
        complex_inputs = torch.stack([
            global_token, heavy_token, light_token, antigen_token, interface_token
        ], dim=1)
        complex_masks = torch.stack([
            valid.any(dim=1), meta["heavy"].any(dim=1), meta["light"].any(dim=1),
            meta["antigen"].any(dim=1), interface_mask.any(dim=1)
        ], dim=1)
        complex_encoded = self.complex_encoder(complex_inputs, complex_masks)
        complex_token = complex_encoded[:, 0]

                                                                         
        pair_hl_mask = hl_valid & meta["antibody"]
        pair_abag_mask = abag_valid
        entity_tokens = [
            complex_token,
            self._masked_global(hl_ctx, pair_hl_mask),
            self._masked_global(abag_ctx, pair_abag_mask),
            self._masked_global(chain_maps[0], meta["heavy"]),
            self._masked_global(chain_maps[1], meta["light"]),
        ]
        entity_masks = [
            valid.any(dim=1), pair_hl_mask.any(dim=1), pair_abag_mask.any(dim=1),
            meta["heavy"].any(dim=1), meta["light"].any(dim=1),
        ]
        for entity_index in range(6):
            region_mask = (meta["region"] == entity_index) & valid
            entity_tokens.append(self._masked_global(region_maps[entity_index], region_mask))
            entity_masks.append(region_mask.any(dim=1))
        entity_tokens = torch.stack(entity_tokens, dim=1)
        entity_masks = torch.stack(entity_masks, dim=1)

                                                                                
        entity_token_maps = [
            (base + complex_token[:, None, :]) * valid.unsqueeze(-1).to(base.dtype),
            hl_ctx, abag_ctx, chain_maps[0], chain_maps[1],
        ] + region_maps
        entity_token_masks = [
            valid, pair_hl_mask, pair_abag_mask, meta["heavy"], meta["light"],
        ] + [((meta["region"] == i) & valid) for i in range(6)]
        entity_token_maps = torch.stack(entity_token_maps, dim=1)
        entity_token_masks = torch.stack(entity_token_masks, dim=1)

                                                                               
                                                                       
                                            
                                                              
        residue_observability = (
            (~generated).to(base.dtype)
            + case_reliability[:, None].to(base.dtype) * generated.to(base.dtype)
        ) * valid.to(base.dtype)
        entity_counts = entity_token_masks.to(base.dtype).sum(dim=-1)
        entity_observability = (
            entity_token_masks.to(base.dtype)
            * residue_observability[:, None, :]
        ).sum(dim=-1) / entity_counts.clamp_min(1.0)
        entity_observability = entity_observability * entity_masks.to(base.dtype)

                                                                                 
                                                                             
                                                                                
                                                
        entity_context = self.entity_rus_encoder(
            entity_tokens, entity_masks, confidence=entity_observability
        )

        chain_local = torch.where(
            meta["heavy"].unsqueeze(-1), chain_maps[0],
            torch.where(meta["light"].unsqueeze(-1), chain_maps[1], torch.zeros_like(base))
        )
        chain_entity = torch.where(
            meta["heavy"].unsqueeze(-1), entity_context[:, 3, None, :],
            torch.where(meta["light"].unsqueeze(-1), entity_context[:, 4, None, :], torch.zeros_like(base))
        )
        chain_ctx = chain_local + chain_entity

        region_local = torch.zeros_like(base)
        region_entity = torch.zeros_like(base)
        for entity_index in range(6):
            select = (meta["region"] == entity_index) & valid
            region_local = torch.where(select.unsqueeze(-1), region_maps[entity_index], region_local)
            region_entity = torch.where(
                select.unsqueeze(-1), entity_context[:, 5 + entity_index, None, :], region_entity
            )
        region_ctx = region_local + region_entity

        complex_ctx = entity_context[:, 0, None, :].expand_as(base)
        hl_ctx = hl_ctx + entity_context[:, 1, None, :]
        abag_ctx = abag_ctx + entity_context[:, 2, None, :]
        raw = torch.stack([complex_ctx, hl_ctx, abag_ctx, chain_ctx, region_ctx], dim=2)
        masks = torch.stack([
            valid, hl_valid, abag_valid, meta["antibody"], meta["region"] >= 0,
        ], dim=2)
        return raw, masks, meta, {
            "hl": hl_ctx,
            "ab_ag": abag_ctx,
            "hl_valid": hl_valid,
            "ab_ag_valid": abag_valid,
            "entity_tokens": entity_tokens,
            "entity_context": entity_context,
            "entity_masks": entity_masks,
            "entity_token_maps": entity_token_maps,
            "entity_token_masks": entity_token_masks,
            "entity_observability": entity_observability,
            "residue_observability": residue_observability,
        }

    def forward(self, node_feat, pair_feat, p_t, beta, mask_generate, mask_res, batch=None):
        if not self.enabled:
            return node_feat, {"enabled": False}

        valid = mask_res.bool()
        generated = mask_generate.bool() & valid
        b, length, _ = node_feat.shape
        time = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_node = self.time_embed(time)[:, None, :].expand(b, length, -1)

                                                                               
                                                                              
                                                     
        clock_gamma = torch.exp(self.semantic_clock_log_gamma)
        beta_unit = beta.clamp(0.0, 1.0)
        interior_beta = beta_unit.clamp(min=1e-7, max=1.0 - 1e-7)
        beta_for_pow = torch.where(
            (beta_unit <= 0.0) | (beta_unit >= 1.0),
            torch.ones_like(interior_beta), interior_beta,
        )
        warped_beta = beta_for_pow.pow(clock_gamma)
        case_reliability = torch.where(
            beta_unit <= 0.0, torch.zeros_like(warped_beta),
            torch.where(beta_unit >= 1.0, torch.ones_like(warped_beta), warped_beta),
        )
        reliability = case_reliability[:, None].expand(b, length)

        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair_mean = (pair_feat * pair_valid.unsqueeze(-1).to(pair_feat.dtype)).sum(dim=2)
        pair_mean = pair_mean / pair_valid.sum(dim=2, keepdim=True).clamp_min(1).to(pair_feat.dtype)
        base = node_feat + self.pair_bias(pair_mean) + time_node

        raw, level_masks, meta, pair_contexts = self._routed_levels(
            base, pair_feat, p_t, batch, valid, generated, case_reliability
        )
                                                                        
                                                                
        levels = raw * level_masks.unsqueeze(-1).to(raw.dtype)

                                                                           
                                                                               
        entity_context = pair_contexts["entity_context"]
        entity_masks = pair_contexts["entity_masks"]
        entity_mask_f = entity_masks.to(entity_context.dtype)
        entity_observability = pair_contexts["entity_observability"]
                                                                              
                                                                             
        entity_evidence = entity_context * entity_observability.unsqueeze(-1)
        entity_flat = entity_evidence.reshape(b, -1)
        entity_rus_input = torch.cat([
            entity_flat, time_node[:, 0], entity_observability.to(entity_context.dtype)
        ], dim=-1)
        shared_case = self.rus_shared(entity_rus_input)
        synergy_case = self.rus_synergy(entity_rus_input)
        entity_unique = torch.stack([
            proj(entity_context[:, i, :])
            for i, proj in enumerate(self.rus_unique)
        ], dim=1) * entity_mask_f.unsqueeze(-1)
        shared = shared_case[:, None, :].expand(b, length, -1)
        synergy = synergy_case[:, None, :].expand(b, length, -1)

        unique_complex = entity_unique[:, 0, None, :].expand(b, length, -1)
        unique_hl = entity_unique[:, 1, None, :].expand(b, length, -1)
        unique_abag = entity_unique[:, 2, None, :].expand(b, length, -1)
        unique_chain = torch.where(
            meta["heavy"].unsqueeze(-1), entity_unique[:, 3, None, :],
            torch.where(meta["light"].unsqueeze(-1), entity_unique[:, 4, None, :], torch.zeros_like(base))
        )
        unique_region = torch.zeros_like(base)
        for entity_index in range(6):
            select = (meta["region"] == entity_index) & valid
            unique_region = torch.where(
                select.unsqueeze(-1), entity_unique[:, 5 + entity_index, None, :], unique_region
            )
        unique = torch.stack([
            unique_complex, unique_hl, unique_abag, unique_chain, unique_region
        ], dim=2)
        unique = unique * level_masks.unsqueeze(-1).to(unique.dtype)

                                                                        
        entity_token_maps = pair_contexts["entity_token_maps"]
        entity_token_masks = pair_contexts["entity_token_masks"]
        entity_messages = torch.stack([
            reader(
                base, entity_token_maps[:, entity_index], entity_unique[:, entity_index],
                entity_token_masks[:, entity_index], valid,
                entity_observability[:, entity_index],
            )
            for entity_index, reader in enumerate(self.constraint_readers)
        ], dim=2)
        complex_message = entity_messages[:, :, 0]
        hl_message = entity_messages[:, :, 1]
        abag_message = entity_messages[:, :, 2]
        chain_message = torch.where(
            meta["heavy"].unsqueeze(-1), entity_messages[:, :, 3],
            torch.where(meta["light"].unsqueeze(-1), entity_messages[:, :, 4], torch.zeros_like(base))
        )
        region_message = torch.zeros_like(base)
        for entity_index in range(6):
            select = (meta["region"] == entity_index) & valid
            region_message = torch.where(
                select.unsqueeze(-1), entity_messages[:, :, 5 + entity_index], region_message
            )
        constraint_messages = torch.stack([
            complex_message, hl_message, abag_message, chain_message, region_message
        ], dim=2)
        constraint_messages = constraint_messages * level_masks.unsqueeze(-1).to(base.dtype)
        constraint_flat = constraint_messages.reshape(b, length, -1)

                                                                               
        self_query = self.self_query(torch.cat([
            base, shared, constraint_flat, time_node
        ], dim=-1))
        self_query = self._safe_unit(self_query)
        prototypes = self._safe_unit(self.role_prototypes)
        self_logits = torch.matmul(self_query, prototypes.transpose(0, 1))
        prior_logits = (self_logits + self.shared_role_mod(shared)
                        + self.unique_role_mod(constraint_flat))
        game_context = self.game_context(torch.cat([
            shared, constraint_flat, synergy, time_node
        ], dim=-1))
                                                                         
                                                                      
                                                                       
        has_generated = generated.any(dim=1, keepdim=True)
        anchor_mask = torch.where(has_generated, generated, valid)
        prior_role_direction = self._case_anchor_role_logits(prior_logits, anchor_mask)
        current_role = F.softmax(reliability.unsqueeze(-1) * prior_role_direction, dim=-1)
        centered_role = current_role - current_role.new_full((), 1.0 / self.num_roles)
                                                                          
                                                                              
                                                                      
        eye = torch.eye(length, device=node_feat.device, dtype=torch.bool)[None]
        neighbor_mask = pair_valid & ~eye
        distance = torch.cdist(p_t.float(), p_t.float()).to(base.dtype)
        distance_weight = neighbor_mask.to(base.dtype)
        distance_scale = (
            (distance * distance_weight).sum(dim=(1, 2), keepdim=True)
            / distance_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        ).detach().clamp_min(torch.finfo(base.dtype).eps)
        physical_logits = -distance / distance_scale + self.pair_neighbor_score(pair_feat).squeeze(-1)
                                                                                 
        s_i = synergy[:, :, None, :].expand(-1, -1, length, -1)
        s_j = synergy[:, None, :, :].expand(-1, length, -1, -1)
        edge_s = self.synergy_edge(torch.cat([
            s_i, s_j, base[:, :, None, :].expand(-1, -1, length, -1)
        ], dim=-1)).squeeze(-1)
        pair_reliability = reliability[:, :, None] * reliability[:, None, :]
        neighbor_logits = physical_logits + pair_reliability * edge_s
        neighbor_weights = self._masked_softmax(neighbor_logits, neighbor_mask, dim=-1)

                                                                             
                                                                            
                                                                             
                                                                             
        base_i = base[:, :, None, :].expand(-1, -1, length, -1)
        base_j = base[:, None, :, :].expand(-1, length, -1, -1)
        game_i = game_context[:, :, None, :].expand(-1, -1, length, -1)
        game_j = game_context[:, None, :, :].expand(-1, length, -1, -1)
        payoff = self.payoff_head(torch.cat([base_i, base_j, game_i, game_j], dim=-1))
        payoff = payoff.reshape(b, length, length, self.num_roles, self.num_roles)
                                                                            
                                                                             
        payoff_rev = self.payoff_head(torch.cat([base_j, base_i, game_j, game_i], dim=-1))
        payoff_rev = payoff_rev.reshape(b, length, length, self.num_roles, self.num_roles)
        payoff = 0.5 * (payoff + payoff_rev.transpose(1, 2).transpose(-1, -2))
                                                                            
                                                                              
                                                                      
        expected_payoff = torch.einsum("bijkr,bjr,bij->bik", payoff, centered_role, neighbor_weights)

                                                                             
                                                                             
        updated_role_direction = self._case_anchor_role_logits(
            prior_role_direction + expected_payoff, anchor_mask
        )
        role_logits = reliability.unsqueeze(-1) * updated_role_direction
        role = F.softmax(role_logits, dim=-1)
        centered_final_role = role - role.new_full((), 1.0 / self.num_roles)

                                                                          
                                                                              
                                                                          
        prior_neighbor_weights = neighbor_weights

                                                                              
                                                                              
                                                                             
        max_centered_norm = ((self.num_roles - 1.0) / self.num_roles) ** 0.5
                                                                         
                                                                           
                                                                         
        amplitude_floor = 1e-12
        role_square = centered_final_role.float().square().sum(dim=-1)
        smooth_norm = (role_square + amplitude_floor).sqrt() - amplitude_floor ** 0.5
        smooth_max = (max_centered_norm ** 2 + amplitude_floor) ** 0.5 - amplitude_floor ** 0.5
        identity_amplitude = (smooth_norm / smooth_max).to(levels.dtype)
        identity_amplitude = identity_amplitude.clamp(0.0, 1.0) * valid.to(levels.dtype)
                                                                           
                                                                              
                                                          
        neighbor = torch.zeros_like(base)
        role_residual = self.role_value(centered_final_role)
        identity_self = torch.matmul(current_role, prototypes)
        identity_game = torch.matmul(role, prototypes)
        adapted = node_feat + reliability.unsqueeze(-1) * role_residual
        adapted = torch.where(valid.unsqueeze(-1), adapted, node_feat)
        self.last = {
            "enabled": True,
            "levels": levels,
            "raw_levels": raw,
            "level_masks": level_masks,
            "entity_levels": pair_contexts["entity_context"],
            "raw_entity_levels": pair_contexts["entity_tokens"],
            "entity_masks": pair_contexts["entity_masks"],
            "entity_token_maps": pair_contexts["entity_token_maps"],
            "entity_token_masks": pair_contexts["entity_token_masks"],
            "entity_observability": pair_contexts["entity_observability"],
            "residue_observability": pair_contexts["residue_observability"],
            "shared": shared,
            "shared_case": shared_case,
            "unique": unique,
            "entity_unique": entity_unique,
            "entity_constraint_messages": entity_messages,
            "constraint_messages": constraint_messages,
                                                                             
                                                                              
                                                                      
            "identity_self": identity_self,
            "identity_game": identity_game,
            "level_global": torch.stack([
                self._masked_global(levels[..., i, :], level_masks[..., i])
                for i in range(len(self.LEVELS))
            ], dim=1),
            "synergy": synergy,
            "role": role,
            "centered_final_role": centered_final_role,
            "role_logits": role_logits,
            "prior_role_direction": prior_role_direction,
            "updated_role_direction": updated_role_direction,
            "self_query": self_query,
            "role_prototypes": prototypes,
            "prior_role_logits": prior_logits,
            "payoff": payoff,
            "expected_payoff": expected_payoff,
            "centered_role": centered_role,
            "game_context": game_context,
            "semantic_reliability": reliability,
            "case_reliability": case_reliability,
            "clock_gamma": clock_gamma,
            "anchor_target_count": anchor_mask.sum(dim=1),
            "identity_amplitude": identity_amplitude,
            "role_residual": role_residual,
            "generated": generated,
            "valid": valid,
            "beta": beta,
            "neighbor_logits": neighbor_logits,
            "neighbor_weights": neighbor_weights,
            "prior_neighbor_weights": prior_neighbor_weights,
            "neighbor_residual": neighbor,
            "hierarchy": {
                **meta,
                "pair_hl": pair_contexts["hl"],
                "pair_ab_ag": pair_contexts["ab_ag"],
                "pair_hl_valid": pair_contexts["hl_valid"],
                "pair_ab_ag_valid": pair_contexts["ab_ag_valid"],
                "same_chain": (meta["chain_type"][:, :, None] == meta["chain_type"][:, None, :]) & pair_valid,
                "same_region": (meta["region"][:, :, None] == meta["region"][:, None, :]) &
                               (meta["region"][:, :, None] >= 0) & (meta["region"][:, None, :] >= 0) & pair_valid,
            },
        }
        return adapted, self.last

    def losses(self, state=None, s0=None, p0=None, R0=None):
        
        if not state or not state.get("enabled", False):
            return None
        zero = state["constraint_messages"].sum() * 0.0
        losses = {"rus": zero}
        generated = state["generated"].bool() & state["valid"].bool()
        level_masks = state["level_masks"].bool()
        case_rho = state["case_reliability"].detach().to(state["constraint_messages"].dtype)
        bsz = generated.size(0)

        def per_case_mean(values, mask):
            mask_f = mask.to(values.dtype)
            while mask_f.dim() < values.dim():
                mask_f = mask_f.unsqueeze(-1)
            reduce_dims = tuple(range(1, values.dim()))
            numerator = (values * mask_f).sum(dim=reduce_dims)
            denominator = mask_f.expand_as(values).sum(dim=reduce_dims).clamp_min(1.0)
            active = mask.reshape(mask.size(0), -1).any(dim=1).to(values.dtype)
            return numerator / denominator, active

        def stage_average(per_case, active):
                                                                          
                                                                   
            return (per_case * active * case_rho).sum() / max(bsz, 1)

        def regression_term(prediction, target, mask):
            values = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
            case, active = per_case_mean(values, mask)
            return stage_average(case, active)

        def classification_term(logits, target, mask):
            values = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
            ).reshape(target.shape)
            case, active = per_case_mean(values, mask)
            return stage_average(case, active)

        hierarchy = state["hierarchy"]
        valid = state["valid"].bool()
        geometry_targets = None
        geometry_masks = None
        partner_properties = None
        valid_seq = None
        own_property = None

        if s0 is not None:
            valid_seq = (s0 >= 0) & (s0 < 20)
            own_property = self.residue_property_index[s0.clamp(0, 19)]

        if p0 is not None:
            pair_valid = valid[:, :, None] & valid[:, None, :]

            def to_residue_frame(vector):
                                                                              
                                                                              
                                                                                   
                if R0 is None:
                    return vector.norm(dim=-1, keepdim=True).expand_as(vector)
                return torch.matmul(R0.transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)

            def nearest_relation(relation):
                relation = relation & pair_valid
                delta = p0[:, :, None, :] - p0[:, None, :, :]
                distance2 = delta.float().square().sum(dim=-1)
                distance2 = distance2.masked_fill(~relation, float("inf"))
                nearest = distance2.argmin(dim=-1)
                gather_xyz = nearest[..., None].expand(-1, -1, 3)
                partner_xyz = p0.gather(1, gather_xyz)
                has_partner = relation.any(dim=-1)
                displacement = torch.where(
                    has_partner[..., None], p0 - partner_xyz, torch.zeros_like(p0)
                )
                displacement = to_residue_frame(displacement)
                partner_aa = None
                if s0 is not None:
                    partner_aa = s0.gather(1, nearest).clamp(0, 19)
                return displacement, has_partner, partner_aa, nearest

                                                                              
                                                                          
            prev_p = torch.roll(p0, shifts=1, dims=1)
            next_p = torch.roll(p0, shifts=-1, dims=1)
            same_prev = torch.roll(hierarchy["chain_type"], 1, 1) == hierarchy["chain_type"]
            same_next = torch.roll(hierarchy["chain_type"], -1, 1) == hierarchy["chain_type"]
            prev_ok = valid & torch.roll(valid, 1, 1) & same_prev
            next_ok = valid & torch.roll(valid, -1, 1) & same_next
            prev_ok[:, 0] = False
            next_ok[:, -1] = False
            local_count = prev_ok.to(p0.dtype) + next_ok.to(p0.dtype)
            local_sum = ((prev_p - p0) * prev_ok[..., None].to(p0.dtype) +
                         (next_p - p0) * next_ok[..., None].to(p0.dtype))
            region_target = to_residue_frame(
                local_sum / local_count.clamp_min(1.0)[..., None]
            )
            region_ok = local_count > 0

                                                                               
                                                                               
            fixed = valid & ~generated & hierarchy["antibody"]
            chain_relation = hierarchy["same_chain"] & fixed[:, None, :]
            chain_target, chain_ok, _, _ = nearest_relation(chain_relation)

            hl_relation = (
                hierarchy["heavy"][:, :, None] & hierarchy["light"][:, None, :]
            ) | (
                hierarchy["light"][:, :, None] & hierarchy["heavy"][:, None, :]
            )
            hl_target, hl_ok, hl_partner_aa, hl_nearest = nearest_relation(hl_relation)

            abag_relation = (
                hierarchy["antibody"][:, :, None] & hierarchy["antigen"][:, None, :]
            ) | (
                hierarchy["antigen"][:, :, None] & hierarchy["antibody"][:, None, :]
            )
            abag_target, abag_ok, abag_partner_aa, abag_nearest = nearest_relation(abag_relation)

                                                                           
                                                                          
            target_count = generated.sum(dim=1, keepdim=True).clamp_min(1).to(p0.dtype)
            target_center = (p0 * generated[..., None].to(p0.dtype)).sum(dim=1, keepdim=True) / target_count[..., None]
            complex_target = to_residue_frame(p0 - target_center)
            complex_ok = generated.any(dim=1, keepdim=True).expand_as(generated)

            geometry_targets = torch.stack([
                complex_target, hl_target, abag_target, chain_target, region_target,
            ], dim=2)
            geometry_masks = torch.stack([
                complex_ok, hl_ok, abag_ok, chain_ok, region_ok,
            ], dim=2) & level_masks

            if s0 is not None:
                partner_properties = {
                    "hl": self.residue_property_index[hl_partner_aa],
                    "abag": self.residue_property_index[abag_partner_aa],
                    "hl_valid": hl_ok & valid_seq & valid_seq.gather(1, hl_nearest),
                    "abag_valid": abag_ok & valid_seq & valid_seq.gather(1, abag_nearest),
                }

                                                                            
                                                                                
                                                                  
        constraint_messages = state["constraint_messages"]
        constraint_sequence_logits = torch.stack([
            head(constraint_messages[..., i, :])
            for i, head in enumerate(self.constraint_sequence_heads)
        ], dim=2)
        constraint_geometry = torch.stack([
            head(constraint_messages[..., i, :])
            for i, head in enumerate(self.constraint_geometry_heads)
        ], dim=2)
        identity_self_sequence_logits = self.identity_self_sequence(state["identity_self"])
        identity_self_geometry = self.identity_self_geometry(state["identity_self"])
        identity_game_sequence_logits = self.identity_game_sequence(state["identity_game"])
        identity_game_geometry = self.identity_game_geometry(state["identity_game"])

                                                                               
                                                                               
                                                                         
        field_terms = []
        if own_property is not None:
            logits = constraint_sequence_logits
            region_mask = generated & level_masks[..., 4] & valid_seq
            field_terms.append(classification_term(logits[..., 4, :], own_property, region_mask))
            if partner_properties is not None:
                field_terms.append(classification_term(
                    logits[..., 1, :], partner_properties["hl"],
                    generated & level_masks[..., 1] & partner_properties["hl_valid"],
                ))
                field_terms.append(classification_term(
                    logits[..., 2, :], partner_properties["abag"],
                    generated & level_masks[..., 2] & partner_properties["abag_valid"],
                ))
        if geometry_targets is not None:
            for level_index in range(len(self.LEVELS)):
                field_terms.append(regression_term(
                    constraint_geometry[..., level_index, :],
                    geometry_targets[..., level_index, :],
                    generated & geometry_masks[..., level_index],
                ))
        field_loss = torch.stack(field_terms).mean() if field_terms else zero
        losses["constraint_field"] = field_loss

                                                                               
                                                                                  
        self_terms = []
        if own_property is not None:
            self_terms.append(classification_term(
                identity_self_sequence_logits, own_property,
                generated & valid_seq,
            ))
        if geometry_targets is not None:
            self_terms.append(regression_term(
                identity_self_geometry[..., :3], geometry_targets[..., 4, :],
                generated & geometry_masks[..., 4],
            ))
            self_terms.append(regression_term(
                identity_self_geometry[..., 3:], geometry_targets[..., 3, :],
                generated & geometry_masks[..., 3],
            ))
        self_loss = torch.stack(self_terms).mean() if self_terms else zero
        losses["identity_self"] = self_loss

                                                                           
                                                                              
                                                             
        game_terms = []
        if partner_properties is not None:
            game_logits = identity_game_sequence_logits
            game_terms.append(classification_term(
                game_logits[..., :self.num_property_classes], partner_properties["hl"],
                generated & partner_properties["hl_valid"],
            ))
            game_terms.append(classification_term(
                game_logits[..., self.num_property_classes:], partner_properties["abag"],
                generated & partner_properties["abag_valid"],
            ))
        if geometry_targets is not None:
            for output_index, level_index in enumerate((1, 2, 0)):
                game_terms.append(regression_term(
                    identity_game_geometry[..., output_index * 3:(output_index + 1) * 3],
                    geometry_targets[..., level_index, :],
                    generated & geometry_masks[..., level_index],
                ))
        game_loss = torch.stack(game_terms).mean() if game_terms else zero
        losses["identity_game"] = game_loss

                                                                             
        entity_context = state["entity_levels"]
        entity_masks = state["entity_masks"].bool()
        shared_case = state["shared_case"]
        projected = [self.rus_vicreg[i](entity_context[:, i]) for i in range(len(self.ENTITY_LEVELS))]
                                                                              
                                                                               
                                                                              
        entity_q = state["entity_observability"].detach().float()
        inv_terms, var_terms, cov_terms = [], [], []
        for i, zi in enumerate(projected):
            weight = entity_q[:, i] * entity_masks[:, i].to(entity_q.dtype)
            if bool((weight > 0).any()):
                per_case_inv = (zi.float() - shared_case.float()).square().mean(dim=-1)
                inv_terms.append((per_case_inv * weight).sum() / weight.sum().clamp_min(1e-8))
            active = weight > 0
            if active.sum() > 1:
                za = zi[active].float()
                wa = weight[active]
                wa_sum = wa.sum().clamp_min(1e-8)
                mean = (za * wa[:, None]).sum(dim=0, keepdim=True) / wa_sum
                centered = za - mean
                variance = (centered.square() * wa[:, None]).sum(dim=0) / wa_sum
                std = torch.sqrt(variance + 1e-4)
                var_terms.append(F.relu(1.0 - std).mean())
                cov = (centered * wa[:, None]).transpose(0, 1) @ centered / wa_sum
                cov = cov - torch.diag(torch.diag(cov))
                cov_terms.append(cov.square().mean())
        inv = torch.stack(inv_terms).mean() if inv_terms else zero
        var_loss = torch.stack(var_terms).mean() if var_terms else zero
        cov_loss = torch.stack(cov_terms).mean() if cov_terms else zero
        losses["vicreg_inv"] = inv
        losses["vicreg_var"] = var_loss
        losses["vicreg_cov"] = cov_loss
        losses["infonce"] = inv                                                     

                                                                                   
        crosscov_terms = []
        entity_unique = state["entity_unique"]
        for i in range(len(self.ENTITY_LEVELS)):
            weight = entity_q[:, i] * entity_masks[:, i].to(entity_q.dtype)
            active = weight > 0
            if active.sum() > 1:
                wa = weight[active]
                wa_sum = wa.sum().clamp_min(1e-8)
                r0 = shared_case[active].float()
                u0 = entity_unique[active, i].float()
                r = r0 - (r0 * wa[:, None]).sum(dim=0, keepdim=True) / wa_sum
                u = u0 - (u0 * wa[:, None]).sum(dim=0, keepdim=True) / wa_sum
                crosscov_terms.append(
                    ((r * wa[:, None]).transpose(0, 1) @ u / wa_sum).square().mean()
                )
        crosscov = torch.stack(crosscov_terms).mean() if crosscov_terms else zero
        losses["crosscov"] = crosscov

        proto = self._safe_unit(self.role_prototypes)
        gram = proto @ proto.transpose(0, 1)
        eye = torch.eye(self.num_roles, device=gram.device, dtype=gram.dtype)
        proto_loss = (gram - eye).square().mean()
        losses["prototype"] = proto_loss

        semantic = field_loss + self_loss + game_loss
        regularization = var_loss + cov_loss + crosscov + proto_loss
        losses["semantic"] = semantic
        losses["regularization"] = regularization
                                                                               
                                                                            
        losses["rus"] = inv + semantic + regularization

        action_mask = generated
        losses["reliability_mean"] = (
            state["semantic_reliability"][action_mask].mean() if action_mask.any() else zero
        )
        losses["metric_clock_gamma"] = state["clock_gamma"].detach()
        losses["metric_entity_observability_mean"] = (
            state["entity_observability"].detach().sum()
            / state["entity_masks"].detach().to(state["entity_observability"].dtype).sum().clamp_min(1.0)
        )
        losses["metric_anchor_target_count"] = state["anchor_target_count"].float().mean().detach()
        if action_mask.any():
            role = state["role"][action_mask].float()
            losses["metric_role_max_probability"] = role.max(dim=-1).values.mean().detach()
            entropy = -(role.clamp_min(1e-12).log() * role).sum(dim=-1)
            losses["metric_role_normalized_entropy"] = (
                entropy / torch.log(role.new_tensor(float(self.num_roles)))
            ).mean().detach()
        else:
            losses["metric_role_max_probability"] = zero.detach()
            losses["metric_role_normalized_entropy"] = zero.detach()
        return losses

def generated_pair_mask(state):
    generated = state["generated"].bool()
    return generated[:, :, None] & generated[:, None, :]





