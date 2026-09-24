import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import functools
from tqdm.auto import tqdm

from moda.modules.common.geometry import apply_rotation_to_vector
from moda.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from moda.modules.encoders.ga import GAEncoder
from moda.modules.cahf.structured_evidence import NativeRiskHead, StructuredEvidenceCoordinator
from moda.modules.cahf.hierarchical_rus_role_adapter import HierarchicalRUSRoleAdapter
from moda.modules.cahf.identity_directed_modal_coupdate import IdentityDirectedModalCoUpdate
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)             
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)             

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')              
    loss = loss.reshape(size + [3]).sum(dim=-1)         
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}, cahf_opt=None):
        super().__init__()
        self.cahf_opt = cahf_opt or {}
        evidence_opt = self.cahf_opt.get("structured_evidence", {})
        risk_opt = self.cahf_opt.get("native_risk", {})
        rus_opt = self.cahf_opt.get("hierarchical_rus_role", {})
        ga_block_opt = encoder_opt.get("ga_block_opt", {}) if isinstance(encoder_opt, dict) else {}
        num_heads = int(ga_block_opt.get("num_heads", 12))
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)                 
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)
        self.evidence = StructuredEvidenceCoordinator(
            pair_feat_dim=pair_feat_dim,
            num_heads=num_heads,
            enabled=bool(evidence_opt.get("enabled", False)),
        )
        self.native_risk_head = NativeRiskHead(
            node_feat_dim=res_feat_dim,
            enabled=bool(risk_opt.get("enabled", False)),
        )
        self.rus_role_adapter = HierarchicalRUSRoleAdapter(
            node_dim=res_feat_dim, pair_dim=pair_feat_dim,
            enabled=bool(rus_opt.get("enabled", False)),
            num_roles=int(rus_opt.get("num_roles", 3)),
        )
        directed_opt = self.cahf_opt.get("identity_directed_modal", {})
        self.identity_directed_modal = IdentityDirectedModalCoUpdate(
            node_dim=res_feat_dim,
            pair_dim=pair_feat_dim,
            hidden_dim=int(directed_opt.get("hidden_dim", res_feat_dim)),
            role_dim=int(directed_opt.get("role_dim", 3)),
            semantic_edge_dim=int(directed_opt.get("semantic_edge_dim", 8)),
        ) if bool(directed_opt.get("enabled", False)) else None
                                                                         
                                                                            
                                                                            
        self.modality_seed = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(res_feat_dim), nn.Linear(res_feat_dim, res_feat_dim, bias=False))
            for name in ("seq", "pos", "rot")
        })
        self.semantic_context_base = nn.Sequential(
            nn.LayerNorm(res_feat_dim * 7),
            nn.Linear(res_feat_dim * 7, res_feat_dim),
            nn.SiLU(),
        )
        self.semantic_context_modal = nn.ModuleDict({
            name: nn.Linear(res_feat_dim, res_feat_dim, bias=False)
            for name in ("seq", "pos", "rot")
        })
        self.modal_output_adapters = nn.ModuleDict({
            name: nn.Linear(res_feat_dim, res_feat_dim, bias=False)
            for name in ("seq", "pos", "rot")
        })
        self.last_cahf = {}
        self.last_sequence_logits = None

        self.eps_crd_net = nn.Sequential(                   
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(           
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(               
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20)
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, cahf_context=None):
        
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t)                              
                                                                            
                        
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)],
                                                 dim=-1))                                                                                                                                                                    
        edge_bias = None
        if cahf_context is not None:
            edge_bias = self.evidence(
                pair_feat,
                cahf_context.get("edge_features"),
                cahf_context.get("edge_mask"),
            )
        if self.training:
                                                                           
                                                                              
                                                                        
                                                     
            def _encoder_forward(R_, p_, x_, pair_, mask_):
                return self.encoder(
                    R_, p_, x_, pair_, mask_, edge_bias=edge_bias
                )

            res_feat = checkpoint(
                _encoder_forward,
                R,
                p_t,
                res_feat,
                pair_feat,
                mask_res,
                use_reentrant=False,
            )
        else:
            res_feat = self.encoder(
                R, p_t, res_feat, pair_feat, mask_res, edge_bias=edge_bias
            )                     
                                                                           
                                                                              
                                                                             
                                                                       
        res_feat, rus_role = self.rus_role_adapter(
            res_feat, pair_feat, p_t, beta, mask_generate, mask_res,
            batch=(cahf_context.get("batch") if cahf_context is not None else None),
        )
        directed_info = None
        seq_feat = pos_feat = rot_feat = res_feat
        if self.identity_directed_modal is not None and rus_role is not None:
            constraint_messages = rus_role["constraint_messages"]
            semantic_input = torch.cat([
                rus_role["shared"],
                constraint_messages.reshape(N, L, -1),
                rus_role["synergy"],
            ], dim=-1)
            semantic_base = self.semantic_context_base(semantic_input)
            hierarchy_contexts = {
                name: self.semantic_context_modal[name](semantic_base)
                for name in ("seq", "pos", "rot")
            }
            modality_states = {
                name: self.modality_seed[name](res_feat)
                for name in ("seq", "pos", "rot")
            }
            receiver_masks = {
                "seq": mask_generate.bool(),
                "pos": mask_generate.bool(),
                "rot": mask_generate.bool(),
            }
            roles = {"generated": mask_generate.bool(), "valid": mask_res.bool()}
            directed_contexts, directed_info = self.identity_directed_modal(
                modality_states=modality_states,
                hierarchy_contexts=hierarchy_contexts,
                role_probability=rus_role["role"],
                p_t=p_t,
                r_t=R,
                pair_feat=pair_feat,
                edge_features=(cahf_context.get("edge_features")
                               if cahf_context is not None else None),
                roles=roles,
                                                                          
                                                                             
                beta=rus_role["case_reliability"],
                receiver_masks=receiver_masks,
            )
            seq_feat = res_feat + self.modal_output_adapters["seq"](directed_contexts["seq"])
            pos_feat = res_feat + self.modal_output_adapters["pos"](directed_contexts["pos"])
            rot_feat = res_feat + self.modal_output_adapters["rot"](directed_contexts["rot"])
        native_risk = self.native_risk_head(res_feat)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)             
        seq_in_feat = torch.cat([seq_feat, t_embed], dim=-1)
        pos_in_feat = torch.cat([pos_feat, t_embed], dim=-1)
        rot_in_feat = torch.cat([rot_feat, t_embed], dim=-1)

                          
        eps_crd = self.eps_crd_net(pos_in_feat)                       
                                                                            
                                                                              
                                                                             
        eps_pos = apply_rotation_to_vector(R, eps_crd)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos,
                              torch.zeros_like(eps_pos))                            

                         
        eps_rot = self.eps_rot_net(rot_in_feat)
        U = so3vec_to_rotation(eps_rot)
        R_next = R @ U                
        v_next = rotation_to_so3vec(R_next)                               
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)                         

                                                
        seq_logits = self.eps_seq_net(seq_in_feat)
        self.last_sequence_logits = seq_logits
        c_denoised = F.softmax(seq_logits, dim=-1)

        self.last_cahf = {
            "edge_bias": edge_bias,
            "native_risk": native_risk,
            "rus_role": rus_role,
            "identity_directed_modal": directed_info,
        }

        return v_next, R_next, eps_pos, c_denoised                                                    


class FullDPM(nn.Module):

    def __init__(
            self,
            res_feat_dim,
            pair_feat_dim,
            num_steps,
            eps_net_opt={},
            trans_rot_opt={},
            trans_pos_opt={},        
            trans_seq_opt={},        
            position_mean=[0.0, 0.0, 0.0],
            position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)             
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)               
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)               
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)               

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale                                    
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence,
                t=None):
        N, L = res_feat.shape[:2]
        if t is None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long,
                              device=self._dummy.device)                               
        p_0 = self._normalize_position(p_0)        

        if denoise_structure:
                                   
            R_0 = so3vec_to_rotation(v_0)                       
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)          
                                    
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)                           
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
                                   
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)                                             
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
                                                                                
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(                                         
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )                                                          

        loss_dict = {}

                       
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)                                    
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)                         
        loss_dict['rot'] = loss_rot

                       
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)                  
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)                  
        loss_dict['pos'] = loss_pos

                                                                 
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)                                               
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)                     
        kldiv = F.kl_div(
            input=log_post_pred,
            target=post_true,
            reduction='none',
            log_target=False
        ).sum(dim=-1)          
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)                  
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
            self,
            v, p, s,
            res_feat, pair_feat,
            mask_generate, mask_res,
            sample_structure=True, sample_sequence=True,
            pbar=False,
    ):
        
        N, L = v.shape[:2]
        p = self._normalize_position(p)

                                                                                       
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}                        
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):              
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )                                      

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)               
            traj[t] = tuple(x.cpu() for x in traj[t])                                                     

        return traj                                                                                                         

    @torch.no_grad()
    def optimize(
            self,
            v, p, s,
            opt_step: int,
            res_feat, pair_feat,
            mask_generate, mask_res,
            sample_structure=True, sample_sequence=True,
            pbar=False,
    ):
        
        N, L = v.shape[:2]
        start_step = int(opt_step)
        if start_step < 1:
            raise ValueError(f'opt_step must be >= 1, got {opt_step!r}.')
                                                                              
                                             
        start_step = min(start_step, int(self.num_steps))
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=start_step, dtype=torch.long, device=self._dummy.device)

                                                                                       
        if sample_structure:
                                   
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
                                    
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {start_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=start_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(start_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )                                      

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])                                       

        return traj

