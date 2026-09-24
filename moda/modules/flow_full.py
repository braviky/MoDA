import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from moda.modules.common.so3 import random_uniform_so3, rotation_to_so3vec, so3vec_to_rotation
from moda.modules.cahf import (
    CaseAdaptiveClock,
    SmoothNashSurplus,
    build_case_features,
    build_structured_edge_features,
)
from moda.modules.diffusion.dpm_full import EpsilonNet

MASK_TOKEN = 20


def rotation_matrix_cosine_loss(R_pred, R_true):
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3
    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)
    ones = torch.ones([ncol], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction="none")
    return loss.reshape(size + [3]).sum(dim=-1)


def _masked_pearson(x, y, mask):
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = mask.reshape(-1) & torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return x.new_zeros(())
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    return torch.where(
        denominator > torch.finfo(x.dtype).eps,
        (x * y).sum() / denominator,
        x.new_zeros(()),
    )


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
        flow_opt=None,
        cahf=None,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.flow_opt = flow_opt or {}
        self.cahf_opt = cahf or {}
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, cahf_opt=self.cahf_opt, **eps_net_opt)
        self.seq_stochasticity = float(self.flow_opt.get("seq_stochasticity", self.flow_opt.get("noise", 0.0)))
        self.seq_temperature = float(self.flow_opt.get("seq_temperature", self.flow_opt.get("temp", 0.1)))
        self.seq_loss_time_weight = bool(self.flow_opt.get("seq_loss_time_weight", False))
        self.ot_coupling_enabled = bool(self.flow_opt.get("ot_coupling", False))
        clock_opt = self.cahf_opt.get("case_clock", {})
        self.case_clock = CaseAdaptiveClock(
            enabled=bool(clock_opt.get("enabled", False)),
            hidden_dim=int(clock_opt.get("hidden_dim", 32)),
            quadrature_nodes=int(clock_opt.get("quadrature_nodes", 16)),
        )
        self.smooth_nash = SmoothNashSurplus(
            enabled=bool(self.cahf_opt.get("cooperative_guidance", {}).get("enabled", False))
        )
        self.register_buffer("position_mean", torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer("position_scale", torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer("_dummy", torch.empty([0]))
        self.debug_context = {}
        self.last_sample_analysis = []
        self.last_sample_analysis_summary = {"enabled": False, "steps": 0}

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def _debug_nonfinite(self, name, tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return False
        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            finite = torch.isfinite(tensor)
            if finite.all():
                return False
            bad = (~finite).sum().item()
            total = tensor.numel()
            finite_vals = tensor[finite]
            if finite_vals.numel() > 0:
                min_val = finite_vals.detach().float().min().item()
                max_val = finite_vals.detach().float().max().item()
            else:
                min_val = float("nan")
                max_val = float("nan")
            ctx = getattr(self, "debug_context", {}) or {}
            print(
                "[FullDPM nonfinite] "
                f"rank={ctx.get('rank', 'NA')} iter={ctx.get('iter', 'NA')} "
                f"stage={name} bad={bad}/{total} finite_min={min_val} finite_max={max_val} "
                f"origin={ctx.get('origin', 'NA')} generate_count={ctx.get('generate_count', 'NA')} "
                f"length={ctx.get('length', 'NA')}",
                flush=True,
            )
            return True
        return False

    def _debug_nonfinite_many(self, **tensors):
        found = False
        for name, tensor in tensors.items():
            found = self._debug_nonfinite(name, tensor) or found
        return found

    @staticmethod
    def _so3_interpolate(v_start, v_end, alpha):
                                                                           
                                                                           
                                                                         
        compute_dtype = torch.promote_types(v_start.dtype, v_end.dtype)
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        v_start = v_start.to(compute_dtype)
        v_end = v_end.to(compute_dtype)
        alpha = alpha.to(compute_dtype)
        R_start = so3vec_to_rotation(v_start)
        R_end = so3vec_to_rotation(v_end)
        rel = torch.matmul(R_start.transpose(-2, -1), R_end)
        rel_vec = rotation_to_so3vec(rel)
        while alpha.dim() < rel_vec.dim():
            alpha = alpha.unsqueeze(-1)
        return rotation_to_so3vec(torch.matmul(R_start, so3vec_to_rotation(alpha * rel_vec)))

    @staticmethod
    def _so3_step_to_target(v_current, v_target, step_size):
        compute_dtype = torch.promote_types(v_current.dtype, v_target.dtype)
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        v_current = v_current.to(compute_dtype)
        v_target = v_target.to(compute_dtype)
        R_current = so3vec_to_rotation(v_current)
        R_target = so3vec_to_rotation(v_target)
        rel = torch.matmul(R_current.transpose(-2, -1), R_target)
        rel_vec = rotation_to_so3vec(rel)
        if not torch.is_tensor(step_size):
            step_size = rel_vec.new_tensor(step_size)
        else:
            step_size = step_size.to(compute_dtype)
        while step_size.dim() < rel_vec.dim():
            step_size = step_size.unsqueeze(-1)
        return rotation_to_so3vec(torch.matmul(R_current, so3vec_to_rotation(step_size * rel_vec)))

    @staticmethod
    def _native_calibrated_position_delta(delta, mask_generate):
        norm = delta.norm(dim=-1)
        rms = torch.sqrt(
            ((norm ** 2) * mask_generate.float()).sum(dim=-1) / mask_generate.float().sum(dim=-1).clamp_min(1.0)
        ).detach()
        factor = (rms[:, None] / norm.clamp_min(1e-8)).clamp_max(1.0)
        return delta * torch.where(mask_generate, factor, torch.ones_like(factor))[:, :, None]

    def _smooth_nash_position_guidance(self, p_next, base_delta, batch, mask_generate, mask_res):
        if (not self.smooth_nash.enabled) or batch is None:
            return p_next
        with torch.enable_grad():
            p_var = p_next.detach().requires_grad_(True)
            utility = self.smooth_nash.utilities(p_var, batch, mask_generate, mask_res)
            phi = torch.log(self.smooth_nash.surplus(utility) + 1e-8).sum(dim=-1).mean()
            grad = torch.autograd.grad(phi, p_var, retain_graph=False, create_graph=False)[0]
        base_rms = torch.sqrt(
            ((base_delta.norm(dim=-1) ** 2) * mask_generate.float()).sum(dim=-1)
            / mask_generate.float().sum(dim=-1).clamp_min(1.0)
        ).detach()
        grad_norm = grad.norm(dim=-1)
        grad_rms = torch.sqrt(
            ((grad_norm ** 2) * mask_generate.float()).sum(dim=-1)
            / mask_generate.float().sum(dim=-1).clamp_min(1.0)
        ).detach()
        guidance = grad * (base_rms / grad_rms.clamp_min(1e-8))[:, None, None]
        guidance = self._native_calibrated_position_delta(guidance, mask_generate)
        return p_next + guidance

    def _compute_case_context(self, batch, mask_generate, mask_res, p_t=None):
        case_features = build_case_features(batch, mask_generate, mask_res)
        edge_features, edge_mask = build_structured_edge_features(batch, mask_generate, mask_res, p_t=p_t)
        return {
            "case_features": case_features,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
        }

    def _clock(self, lam, case_context):
        features = None if case_context is None else case_context.get("case_features")
        return self.case_clock(lam, features)

    def _minibatch_ot_reorder(self, p_noise, v_noise, p_0, mask_generate):
        if (not self.ot_coupling_enabled) or linear_sum_assignment is None or p_0.size(0) <= 1:
            return p_noise, v_noise
        n = p_0.size(0)
        assignment = torch.arange(n, device=p_0.device)
        gen_count = mask_generate.long().sum(dim=-1)
        for count in gen_count.unique():
            group = torch.nonzero(gen_count == count, as_tuple=False).flatten()
            if group.numel() <= 1 or int(count.item()) <= 0:
                continue
            cost = p_0.new_zeros(group.numel(), group.numel())
            for row, i in enumerate(group.tolist()):
                mask_i = mask_generate[i]
                target = p_0[i, mask_i]
                for col, j in enumerate(group.tolist()):
                    source = p_noise[j, mask_i]
                    cost[row, col] = ((target - source) ** 2).sum(dim=-1).mean()
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            assignment[group[row_ind]] = group[col_ind]
        return p_noise[assignment], v_noise[assignment]

    def _make_flow_state(self, v_0, p_0, s_0, mask_generate, tau):
        N, L = s_0.shape
        tau_pos = tau["pos"][:, None].expand(N, L)
        tau_rot = tau["rot"][:, None].expand(N, L)
        tau_seq = tau["seq"][:, None].expand(N, L)
        tau_pos3 = tau_pos[:, :, None]

        p_noise = torch.randn_like(p_0)
        v_noise = random_uniform_so3([N, L], device=p_0.device)
        p_noise, v_noise = self._minibatch_ot_reorder(p_noise, v_noise, p_0, mask_generate)
        p_t = (1.0 - tau_pos3) * p_noise + tau_pos3 * p_0
        pos_velocity = p_0 - p_noise
        p_t = torch.where(mask_generate[:, :, None], p_t, p_0)
        pos_velocity = torch.where(mask_generate[:, :, None], pos_velocity, torch.zeros_like(pos_velocity))

        v_t = self._so3_interpolate(v_noise, v_0, tau_rot)
        v_t = torch.where(mask_generate[:, :, None], v_t, v_0)

        visible = torch.rand(N, L, device=s_0.device) < tau_seq
        s_t = torch.full_like(s_0, MASK_TOKEN)
        s_t = torch.where(visible, s_0, s_t)
        s_t = torch.where(mask_generate, s_t, s_0)
        return v_t, p_t, s_t, pos_velocity

    @staticmethod
    def _masked_sequence_loss(
        seq_logits,
        s_0,
        s_t,
        mask_generate,
        t=None,
        use_time_weight=False,
        cdr_flag=None,
    ):
        loss_mask = mask_generate & (s_t == MASK_TOKEN)
        n, l = s_0.shape
        ce = F.cross_entropy(
            seq_logits.transpose(1, 2),
            s_0.clamp(0, 19),
            reduction="none",
        )
        masked_count = loss_mask.float().sum(dim=-1)
        valid_case = masked_count > 0
        if not valid_case.any():
            return seq_logits.sum() * 0.0
        denom = masked_count.clamp_min(1.0)
        loss = (ce * loss_mask.float()).sum(dim=-1) / denom
        if cdr_flag is not None:
                                                                              
                                                                           
                                                            
            cdr_losses = []
            cdr_present = []
            for cdr_id in range(1, 7):
                region_mask = loss_mask & (cdr_flag == cdr_id)
                region_count = region_mask.float().sum(dim=-1)
                region_loss = (ce * region_mask.float()).sum(dim=-1) / region_count.clamp_min(1.0)
                cdr_losses.append(region_loss)
                cdr_present.append(region_count > 0)
            cdr_losses = torch.stack(cdr_losses, dim=-1)
            cdr_present = torch.stack(cdr_present, dim=-1)
            case_count = cdr_present.sum(dim=-1)
            case_loss = (cdr_losses * cdr_present.to(cdr_losses.dtype)).sum(dim=-1)
            loss = case_loss / case_count.clamp_min(1).to(case_loss.dtype)
        if use_time_weight:
            if t is None:
                raise ValueError("t is required when use_time_weight=True for masked sequence loss")
            loss = loss / (1.0 - t.float()).clamp_min(1e-3)
        return loss[valid_case].mean()

    @staticmethod
    def _multiflow_masked_step(
        c_denoised,
        s_cur,
        mask_generate,
        tau,
        dtau,
        stochasticity,
        temperature,
        local_rate=None,
    ):
        
        logits = torch.log(c_denoised.clamp_min(1e-8)) / max(temperature, 1e-6)
        x1_probs = F.softmax(logits, dim=-1)
        s_next = s_cur.clone()
        masked = mask_generate & (s_cur == MASK_TOKEN)
        n, l = s_cur.shape

        def _batch_time(value):
            if torch.is_tensor(value):
                value = value.to(device=s_cur.device, dtype=x1_probs.dtype).reshape(-1)
            else:
                value = torch.full((n,), float(value), device=s_cur.device, dtype=x1_probs.dtype)
            if value.numel() == 1:
                value = value.expand(n)
            if value.numel() != n:
                raise ValueError(f"Expected one CTMC time per sample, got {value.numel()} for batch {n}")
            return value

        tau = _batch_time(tau).clamp(0.0, 1.0)
        dtau = _batch_time(dtau).clamp_min(0.0)
        final_case = tau + dtau >= 1.0 - 1e-8

        if local_rate is None:
            local_rate = torch.ones_like(s_cur, dtype=x1_probs.dtype)
        tau_grid = tau[:, None]
        dtau_grid = dtau[:, None]
        intensity = local_rate * (1.0 + stochasticity * tau_grid) / (1.0 - tau_grid).clamp_min(1e-6)
        unmask_prob = 1.0 - torch.exp(-dtau_grid * intensity.clamp_min(0.0))
        unmask_prob = torch.where(final_case[:, None], torch.ones_like(unmask_prob), unmask_prob)
        aa_samples = torch.multinomial(x1_probs.reshape(n * l, 20) + 1e-8, 1).view(n, l)
        jump = (torch.rand_like(unmask_prob) < unmask_prob) & masked
        s_next = torch.where(jump, aa_samples, s_next)

        if stochasticity > 0.0:
            remask_prob = (dtau * stochasticity).clamp(0.0, 1.0)[:, None]
            remask = (torch.rand_like(s_cur.float()) < remask_prob) & mask_generate
            remask = remask & (~final_case[:, None])
            s_next = torch.where(remask, torch.full_like(s_next, MASK_TOKEN), s_next)
        return s_next

    def forward(
        self,
        v_0,
        p_0,
        s_0,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        denoise_structure,
        denoise_sequence,
        t=None,
        batch=None,
    ):
        N, L = res_feat.shape[:2]
        if t is None:
            t = torch.rand(N, device=self._dummy.device).clamp(1e-3, 1.0 - 1e-3)
        else:
            if torch.is_floating_point(t):
                t = t.float()
                if t.detach().amax() > 1.0:
                    t = t / float(self.num_steps)
            else:
                t = t.float() / float(self.num_steps)
            t = t.clamp(1e-3, 1.0 - 1e-3)

        p_0 = self._normalize_position(p_0)
        R_0 = so3vec_to_rotation(v_0)
        self._debug_nonfinite_many(
            v_0=v_0,
            p_0_normalized=p_0,
            R_0=R_0,
            res_feat=res_feat,
            pair_feat=pair_feat,
            t=t,
        )

        case_context = self._compute_case_context(batch, mask_generate, mask_res)
        tau = self._clock(t, case_context)
        v_t, p_t, s_t, pos_velocity = self._make_flow_state(v_0, p_0, s_0, mask_generate, tau)
        case_context = self._compute_case_context(batch, mask_generate, mask_res, p_t=p_t)
        case_context["tau"] = tau
        case_context["lambda"] = t
        case_context["batch"] = batch
        case_context["compute_coupling_targets"] = True
        self._debug_nonfinite_many(v_t=v_t, p_t=p_t, pos_velocity_target=pos_velocity)
        if not denoise_structure:
            v_t, p_t = v_0.clone(), p_0.clone()
            pos_velocity = torch.zeros_like(p_0)

        if not denoise_sequence:
            s_t = s_0.clone()

        v_pred, R_pred, pos_velocity_pred, c_denoised = self.eps_net(
            v_t, p_t, s_t, res_feat, pair_feat, t, mask_generate, mask_res, cahf_context=case_context
        )
        self._debug_nonfinite_many(
            v_pred=v_pred,
            R_pred=R_pred,
            pos_velocity_pred=pos_velocity_pred,
            c_denoised=c_denoised,
        )
        seq_logits = getattr(self.eps_net, "last_sequence_logits", None)
        if seq_logits is None:
            raise RuntimeError("EpsilonNet did not expose sequence logits for cross-entropy")
        per_residue_rot_risk = rotation_matrix_cosine_loss(R_pred, R_0)
        per_residue_pos_risk = F.mse_loss(
            pos_velocity_pred, pos_velocity, reduction="none"
        ).sum(dim=-1)
        per_residue_seq_risk = F.cross_entropy(
            seq_logits.transpose(1, 2),
            s_0.clamp(0, 19),
            reduction="none",
        )

        loss_dict = {}
        generated_weight = mask_generate.to(p_t.dtype)
        generated_count = generated_weight.sum().clamp_min(1.0)
        pos_error_physical = (pos_velocity_pred - pos_velocity) * self.position_scale
        pos_velocity_rmse = torch.sqrt(
            (pos_error_physical.square().sum(dim=-1) * generated_weight).sum()
            / (generated_count * pos_error_physical.size(-1))
        )
        p_x0_pred = p_t + (1.0 - tau["pos"])[:, None, None] * pos_velocity_pred
        x0_error_physical = (p_x0_pred - p_0) * self.position_scale
        x0_rmse = torch.sqrt(
            (x0_error_physical.square().sum(dim=-1) * generated_weight).sum()
            / (generated_count * x0_error_physical.size(-1))
        )
        relative_rotation = torch.matmul(R_pred.transpose(-2, -1), R_0)
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        rotation_cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        rotation_angle_deg = torch.acos(rotation_cos) * (180.0 / math.pi)
        coalition_type = None if batch is None else batch.get("coalition_type")
        if torch.is_tensor(coalition_type):
            coalition_type = coalition_type.to(device=p_t.device)
            for type_id, type_name in ((0, "all"), (2, "h3_only"), (3, "subset"), (4, "h3_l1")):
                case_mask = coalition_type == type_id
                loss_dict[f"metric_coalition_{type_name}_fraction"] = (
                    case_mask.to(p_t.dtype).mean().detach()
                )
                residue_mask = mask_generate.bool() & case_mask[:, None]
                residue_weight = residue_mask.to(p_t.dtype)
                residue_count = residue_weight.sum().clamp_min(1.0)
                loss_dict[f"metric_coalition_{type_name}_rot_risk"] = (
                    per_residue_rot_risk * residue_weight
                ).sum().detach() / residue_count
                loss_dict[f"metric_coalition_{type_name}_pos_risk"] = (
                    per_residue_pos_risk * residue_weight
                ).sum().detach() / residue_count
                coalition_sequence_mask = residue_mask & (s_t == MASK_TOKEN)
                coalition_sequence_weight = coalition_sequence_mask.to(p_t.dtype)
                coalition_sequence_count = coalition_sequence_weight.sum().clamp_min(1.0)
                loss_dict[f"metric_coalition_{type_name}_seq_nll"] = (
                    per_residue_seq_risk * coalition_sequence_weight
                ).sum().detach() / coalition_sequence_count
        if denoise_structure:
            loss_dict["rot"] = (per_residue_rot_risk * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
            loss_dict["pos"] = (per_residue_pos_risk * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
            loss_dict["metric_pos_velocity_rmse_angstrom"] = pos_velocity_rmse.detach()
            loss_dict["metric_pos_x0_rmse_angstrom"] = x0_rmse.detach()
            loss_dict["metric_rot_geodesic_deg"] = (
                (rotation_angle_deg * generated_weight).sum() / generated_count
            ).detach()
            if batch is not None and "cdr_flag" in batch:
                for name, cdr_id in (
                    ("h1", 1), ("h2", 2), ("h3", 3),
                    ("l1", 4), ("l2", 5), ("l3", 6),
                ):
                    structure_region = mask_generate.bool() & (batch["cdr_flag"] == cdr_id)
                    structure_count = structure_region.sum().clamp_min(1).to(p_t.dtype)
                    structure_weight = structure_region.to(p_t.dtype)
                    region_x0_rmse = torch.sqrt(
                        (x0_error_physical.square().sum(dim=-1) * structure_weight).sum()
                        / (structure_count * x0_error_physical.size(-1))
                    )
                    region_rot_angle = (
                        rotation_angle_deg * structure_weight
                    ).sum() / structure_count
                    loss_dict[f"metric_pos_{name}_x0_rmse_angstrom"] = region_x0_rmse.detach()
                    loss_dict[f"metric_rot_{name}_geodesic_deg"] = region_rot_angle.detach()
            self._debug_nonfinite_many(
                loss_rot=per_residue_rot_risk,
                loss_pos=per_residue_pos_risk,
                loss_rot_reduced=loss_dict["rot"],
                loss_pos_reduced=loss_dict["pos"],
            )
        else:
            zero = pos_velocity_pred.sum() * 0.0
            loss_dict["rot"] = zero
            loss_dict["pos"] = zero

        if denoise_sequence:
            loss_dict["seq"] = self._masked_sequence_loss(
                seq_logits,
                s_0,
                s_t,
                mask_generate,
                t=t,
                use_time_weight=self.seq_loss_time_weight,
                cdr_flag=None if batch is None else batch.get("cdr_flag"),
            )
            sequence_mask = mask_generate.bool() & (s_t == MASK_TOKEN)
            sequence_count = sequence_mask.sum().clamp_min(1)
            sequence_nll = (
                per_residue_seq_risk * sequence_mask.to(per_residue_seq_risk.dtype)
            ).sum() / sequence_count.to(per_residue_seq_risk.dtype)
            sequence_correct = seq_logits.argmax(dim=-1) == s_0.clamp(0, 19)
            loss_dict["metric_seq_accuracy"] = (
                sequence_correct & sequence_mask
            ).sum().to(c_denoised.dtype) / sequence_count.to(c_denoised.dtype)
            loss_dict["metric_seq_perplexity"] = torch.exp(sequence_nll.detach())
            generated_count = mask_generate.sum().clamp_min(1)
            loss_dict["metric_seq_mask_fraction"] = sequence_mask.sum().to(
                c_denoised.dtype
            ) / generated_count.to(c_denoised.dtype)
            if batch is not None and "cdr_flag" in batch:
                for name, cdr_id in (
                    ("h1", 1), ("h2", 2), ("h3", 3),
                    ("l1", 4), ("l2", 5), ("l3", 6),
                ):
                    region_mask = sequence_mask & (batch["cdr_flag"] == cdr_id)
                    region_count = region_mask.sum()
                    region_nll = (
                        per_residue_seq_risk * region_mask.to(per_residue_seq_risk.dtype)
                    ).sum() / region_count.clamp_min(1).to(per_residue_seq_risk.dtype)
                    loss_dict[f"metric_seq_{name}_nll"] = region_nll.detach()
                    loss_dict[f"metric_seq_{name}_count"] = region_count.to(c_denoised.dtype)
                    region_accuracy = (
                        (sequence_correct & region_mask).sum().to(c_denoised.dtype)
                        / region_count.clamp_min(1).to(c_denoised.dtype)
                    )
                    loss_dict[f"metric_seq_{name}_accuracy"] = region_accuracy.detach()
            self._debug_nonfinite("loss_seq_reduced", loss_dict["seq"])
        else:
            loss_dict["seq"] = c_denoised.sum() * 0.0
        risk_pred = getattr(self.eps_net, "last_cahf", {}).get("native_risk")
        if risk_pred is not None:
            with torch.no_grad():
                risk_target = torch.stack(
                    [
                        per_residue_seq_risk,
                        per_residue_pos_risk,
                        per_residue_rot_risk,
                    ],
                    dim=-1,
                ).detach()
            risk_loss = F.smooth_l1_loss(torch.log1p(risk_pred), torch.log1p(risk_target), reduction="none").sum(dim=-1)
            loss_dict["cahf_value"] = (risk_loss * mask_generate.float()).sum() / (mask_generate.sum().float() + 1e-8)
        rus_role_state = getattr(self.eps_net, "last_cahf", {}).get("rus_role")
        rus_role_losses = self.eps_net.rus_role_adapter.losses(rus_role_state, s0=s_0, p0=p_0, R0=R_0)
        if rus_role_losses is not None:
            loss_dict["semantic_rus_role"] = rus_role_losses["rus"]
            for name, value in rus_role_losses.items():
                if name != "rus":
                    loss_dict[f"metric_semantic_{name}"] = value.detach()

        if self.smooth_nash.enabled and denoise_structure:
            tau_left = (1.0 - tau["pos"])[:, None, None]
            p_x0_pred = p_t + tau_left * pos_velocity_pred
            loss_dict["cahf_nash"] = self.smooth_nash.loss(p_x0_pred, p_0, batch, mask_generate, mask_res)
                                                                          
                                                                              
                                                                             
        return loss_dict

    def sample(
        self,
        v,
        p,
        s,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        sample_structure=True,
        sample_sequence=True,
        pbar=False,
        batch=None,
        **unused_sample_opt,
    ):
        N, L = v.shape[:2]
        num_steps = int(unused_sample_opt.pop("num_steps", self.num_steps))
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        async_mode = unused_sample_opt.pop("async_mode", None)
        if async_mode is not None:
            raise ValueError(
                "async_mode belongs to the legacy discrete diffusion sampler and is not "
                "implemented by this flow sampler. Directed residue-level coupling changes "
                "which evidence each receiver uses; it does not replace the base path clock."
            )
        if unused_sample_opt:
            unknown = ", ".join(sorted(unused_sample_opt))
            raise TypeError(f"Unsupported flow sampling options: {unknown}")
        p = self._normalize_position(p)

        if sample_structure:
            v_t = torch.where(mask_generate[:, :, None], random_uniform_so3([N, L], device=self._dummy.device), v)
            p_t = torch.where(mask_generate[:, :, None], torch.randn_like(p), p)
        else:
            v_t, p_t = v, p

        if sample_sequence:
            s_t = torch.where(mask_generate, torch.full_like(s, MASK_TOKEN), s)
        else:
            s_t = s

        traj = {num_steps: (v_t, self._unnormalize_position(p_t), s_t)}
        self.last_sample_analysis = []
        self.last_sample_analysis_summary = {"enabled": False, "steps": 0}
        piter = functools.partial(tqdm, total=num_steps, desc="Flow sampling") if pbar else (lambda x: x)
        base_context = self._compute_case_context(batch, mask_generate, mask_res)

        for step in piter(range(num_steps, 0, -1)):
            v_cur, p_cur, s_cur = traj[step]
            p_cur = self._normalize_position(p_cur)
            progress = float(num_steps - step) / float(num_steps)
            next_progress = float(num_steps - step + 1) / float(num_steps)
            t = torch.full((N,), progress, device=self._dummy.device)
            t_next = torch.full((N,), next_progress, device=self._dummy.device)
            tau = self._clock(t, base_context)
            tau_next = self._clock(t_next, base_context)
            dtau_pos = (tau_next["pos"] - tau["pos"]).clamp_min(0.0)
            dtau_rot = (tau_next["rot"] - tau["rot"]).clamp_min(0.0)
            dtau_seq = (tau_next["seq"] - tau["seq"]).clamp_min(0.0)
            case_context = self._compute_case_context(batch, mask_generate, mask_res, p_t=p_cur)
            case_context["tau"] = tau
            case_context["lambda"] = t
            case_context["batch"] = batch
            case_context["compute_coupling_targets"] = False
            with torch.no_grad():
                v_target, _, pos_velocity, c_denoised = self.eps_net(
                    v_cur, p_cur, s_cur, res_feat, pair_feat, t, mask_generate, mask_res, cahf_context=case_context
                )
                sequence_rate = None
                rotation_rate = torch.ones_like(mask_generate, dtype=pos_velocity.dtype)
                controlled_velocity = pos_velocity
                                                                                       
                                                                                        
                                                                                      
                                                                   
                pos_delta = dtau_pos[:, None, None] * controlled_velocity
                p_next = p_cur + pos_delta
                so3_step_size = (
                    dtau_rot / (1.0 - tau["rot"]).clamp_min(1e-3)
                )[:, None] * rotation_rate
                v_next = self._so3_step_to_target(v_cur, v_target, so3_step_size)
                s_next = self._multiflow_masked_step(
                    c_denoised,
                    s_cur,
                    mask_generate,
                    tau=tau["seq"],
                    dtau=dtau_seq,
                    stochasticity=self.seq_stochasticity,
                    temperature=self.seq_temperature,
                    local_rate=sequence_rate,
                )
            p_next = self._smooth_nash_position_guidance(p_next, pos_delta, batch, mask_generate, mask_res)

            if not sample_structure:
                v_next, p_next = v_cur, p_cur
            if not sample_sequence:
                s_next = s_cur

            v_next = torch.where(mask_generate[:, :, None], v_next, v)
            p_next = torch.where(mask_generate[:, :, None], p_next, p)
            s_next = torch.where(mask_generate, s_next, s)

            traj[step - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[step] = tuple(x.cpu() for x in traj[step])
                                                                              
                                                                            
        self.last_sample_analysis_summary = {
            "enabled": False,
            "steps": 0,
            "updates": num_steps,
            "reason": "sampling_diagnostics_disabled",
        }
        return traj

    @torch.no_grad()
    def optimize(
        self,
        v,
        p,
        s,
        opt_step,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        sample_structure=True,
        sample_sequence=True,
        pbar=False,
        batch=None,
    ):
        opt_step = int(opt_step)
        if opt_step <= 0:
            raise ValueError(f"opt_step must be positive, got {opt_step}")
                                                                            
                                                                        
                                                                           
        return self.sample(
            v,
            p,
            s,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
            sample_structure=sample_structure,
            sample_sequence=sample_sequence,
            pbar=pbar,
            batch=batch,
            num_steps=opt_step,
        )


