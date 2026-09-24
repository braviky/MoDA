
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from moda.datasets import get_dataset
from moda.models import get_model
from moda.utils.data import PaddingCollate
from moda.utils.misc import get_logger, load_config, seed_all
from moda.utils.train import ValidationLossTape, compute_core_loss, recursive_to, sum_weighted_losses


def _resolve_checkpoint(checkpoint_dir):
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.isdir(checkpoint_dir):
        base_dir = os.path.dirname(checkpoint_dir)
        tag = os.path.basename(checkpoint_dir.rstrip(os.sep))
    else:
        tag_dir = os.path.dirname(checkpoint_dir)
        base_dir = os.path.dirname(tag_dir)
        tag = os.path.basename(tag_dir)
    return base_dir, tag


def _load_zero_checkpoint(model, checkpoint_dir, logger):
    base_dir, tag = _resolve_checkpoint(checkpoint_dir)
    model_state_path = os.path.join(
        os.path.abspath(checkpoint_dir), 'mp_rank_00_model_states.pt'
    )
    optim_state_paths = []
    if os.path.isdir(checkpoint_dir):
        optim_state_paths = [
            name for name in os.listdir(checkpoint_dir)
            if name.endswith('_optim_states.pt')
        ]
    if os.path.isfile(model_state_path) and not optim_state_paths:
        logger.info(f'Loading compact DeepSpeed model state: {model_state_path}')
        payload = torch.load(model_state_path, map_location='cpu')
        state_dict = payload.get('module', payload)
        iteration = payload.get('iteration', int(tag) if str(tag).isdigit() else tag)
    else:
        logger.info(f'Loading ZeRO checkpoint: base={base_dir}, tag={tag}')
        state_dict = get_fp32_state_dict_from_zero_checkpoint(base_dir, tag=tag)
        iteration = int(tag) if str(tag).isdigit() else tag
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if not all(k.startswith('module.') for k in state_dict.keys()):
            raise
        logger.info('Retrying checkpoint load after stripping "module." prefixes')
        stripped = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=True)
    return iteration


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return float(value.detach().float().mean().cpu())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_dir, tag = _resolve_checkpoint(args.checkpoint_dir)
    logger = get_logger(f'val_async_{tag}', args.output_dir)
    logger.info(f'=== async validation start: tag={tag} ===')
    logger.info(f'checkpoint_dir={args.checkpoint_dir}')
    logger.info(f'config={args.config}')
    logger.info(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "")}')

    config, _ = load_config(args.config)
    seed_all(config.train.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}')

    model = get_model(config.model)
    iteration = _load_zero_checkpoint(model, args.checkpoint_dir, logger)
    model = model.to(device)
    model.eval()

    started = time.time()
    val_t_steps = list(getattr(config.train, 'val_t_steps', [10, 30, 50, 70, 90]))
    view_avg = {}
    view_by_t = {}
    for view_index, (view_name, view_cfg) in enumerate(
        (('h3_only', config.dataset.val), ('all_cdr', config.dataset.val_all))
    ):
        val_dataset = get_dataset(view_cfg)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(),
            num_workers=args.num_workers,
        )
        loss_tape = ValidationLossTape()
        diagnostics_by_t = {
            int(t_step): {
                key: []
                for key in (
                    'seq',
                    'pos',
                    'rot',
                    'metric_seq_accuracy',
                    'metric_seq_h3_nll',
                    'metric_seq_h3_accuracy',
                    'metric_pos_x0_rmse_angstrom',
                    'metric_pos_h3_x0_rmse_angstrom',
                    'metric_rot_geodesic_deg',
                    'metric_rot_h3_geodesic_deg',
                )
            }
            for t_step in val_t_steps
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc=f'Validate({tag}/{view_name})', dynamic_ncols=True)
            ):
                batch = recursive_to(batch, device=device)
                batch_size = batch['aa'].shape[0]
                for t_step in val_t_steps:
                    seed_all(
                        int(config.train.seed)
                        + 100000
                        + view_index * 1000000
                        + batch_idx * 100
                        + int(t_step)
                    )
                    fixed_t = torch.full(
                        (batch_size,), int(t_step), dtype=torch.long, device=device
                    )
                    loss_dict = model(batch, t=fixed_t)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict = dict(loss_dict)
                    loss_dict['overall'] = loss
                    loss_dict['core'] = compute_core_loss(loss_dict)
                    loss_tape.update(loss_dict, batch_size)
                    for key, values in diagnostics_by_t[int(t_step)].items():
                        if key in loss_dict:
                            values.append(float(loss_dict[key].detach().cpu()))
        view_avg[view_name], _ = loss_tape.compute_avg()
        view_by_t[view_name] = {
            str(t_step): {
                key: sum(values) / len(values)
                for key, values in metrics.items()
                if values
            }
            for t_step, metrics in diagnostics_by_t.items()
        }
        for key in next(iter(diagnostics_by_t.values())):
            summary = ' '.join(
                f't{t_step}={sum(metrics[key]) / len(metrics[key]):.4f}'
                for t_step, metrics in diagnostics_by_t.items()
                if metrics[key]
            )
            logger.info(f'[val-by-t/{view_name}/{key}] {summary}')

    avg_loss_dict = {
        key: sum(view[key] for view in view_avg.values()) / len(view_avg)
        for key in view_avg['h3_only'].keys()
    }
    elapsed = time.time() - started
    avg_loss = float(avg_loss_dict['overall'].detach().cpu())
    avg_core_loss = float(avg_loss_dict['core'].detach().cpu())

    logger.info(
        '[val_async] Iter %s | loss %.4f | core %.4f | %s | elapsed %.1fs'
        % (
            iteration,
            avg_loss,
            avg_core_loss,
            ' | '.join(
                f'{k} {float(v.detach().cpu()):.4f}'
                for k, v in avg_loss_dict.items()
                if k != 'overall'
            ),
            elapsed,
        )
    )

    result = {
        'iteration': iteration,
        'checkpoint_base': base_dir,
        'checkpoint_tag': tag,
        'avg_loss': avg_loss,
        'avg_core_loss': avg_core_loss,
        'losses': _to_jsonable(avg_loss_dict),
        'losses_by_view': _to_jsonable(view_avg),
        'diagnostics_by_view_time': view_by_t,
        'elapsed_sec': elapsed,
    }
    result_path = os.path.join(args.output_dir, f'val_{tag}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'wrote {result_path}')
    logger.info(f'=== async validation complete: tag={tag} ===')


if __name__ == '__main__':
    main()
