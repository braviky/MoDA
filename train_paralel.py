import os
import sys
import json
                                            

                 
                                   
                                                  
                                                         
 
                                                  
                                                          
                                              
                                         
                                             
                                           
                                                  
                                        
                                   
                                                                                      
                                   
                                   
                                                                                            
 
                       
                                                  
                                                              

import shutil
import argparse
import subprocess
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
                                                                     
from moda.datasets import get_dataset
from moda.models import get_model
from moda.utils.misc import *
from moda.utils.data import *
from moda.utils.train import *
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
local_rank = os.environ.get('LOCAL_RANK', 'N/A')
print(f"[{os.uname()[1]}] PID: {os.getpid()} LOCAL_RANK: {local_rank} CUDA_VISIBLE_DEVICES: {visible_devices}",
      file=sys.stderr, flush=True)
                    
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import deepspeed
import torch.distributed as dist            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resuming from checkpoint. Pass path to tag folder (e.g., .../checkpoints/25000) or a file (e.g., .../25000/mp_rank_00_model_states.pt)')
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--deepspeed_config', type=str, default='./configs/ds_config/gpu8/stage0_bs8_acc1.json')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--val_async', action='store_true', default=False,
                        help='Save checkpoints at val_freq and launch validation in a background process.')
    parser.add_argument('--val_async_gpu', type=str, default=None,
                        help='CUDA_VISIBLE_DEVICES value for async validation. Defaults to the last visible training GPU.')
    parser.add_argument('--val_async_batch_size', type=int, default=5,
                        help='Batch size used by validate_async.py.')
    parser.add_argument('--val_async_num_workers', type=int, default=0,
                        help='DataLoader workers used by validate_async.py.')

    args = parser.parse_args()
    if args.resume is not None and args.finetune is not None:
        parser.error('--resume and --finetune are mutually exclusive; choose one checkpoint mode.')

                                                                           
    global_rank = int(os.environ.get('RANK', 0))
    is_global_main_process = (global_rank == 0)

                                           
    if args.local_rank != -1 and not dist.is_initialized():
        deepspeed.init_distributed()
                              
        global_rank = dist.get_rank()
        is_global_main_process = (global_rank == 0)
                                                          

                  
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

                                                      
    log_dir = None
    ckpt_dir = None

    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
                           
        if args.resume:
            if os.path.isdir(args.resume):
                log_dir = os.path.dirname(os.path.dirname(args.resume))
            else:
                log_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.resume)))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

                                              
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        config_path = os.path.join(log_dir, os.path.basename(args.config))
        if not os.path.exists(config_path):
            shutil.copyfile(args.config, config_path)

    print(f"args.debug {args.debug}")
    print(f"log dir: {log_dir}\ncpkt dir: {ckpt_dir}")
    args.output_dir = log_dir

    if is_global_main_process:
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)

        logger.info(args)
        logger.info(config)
    else:
                                          
        logger = get_logger('train', None)
        writer = BlackHole()

          
    train_dataset = get_dataset(config.dataset.train)
    val_datasets = {
        'h3_only': get_dataset(config.dataset.val),
        'all_cdr': get_dataset(config.dataset.val_all),
    }
    if hasattr(config.dataset, 'val_h3_l1'):
        val_datasets['h3_l1'] = get_dataset(config.dataset.val_h3_l1)
    val_dataset = val_datasets['h3_only']
    val_summary = ' | '.join(
        f'Val {name} {len(dataset)}' for name, dataset in val_datasets.items()
    )
    if is_global_main_process:
        logger.info('Loading dataset...')
        logger.info('Train %d | %s' % (len(train_dataset), val_summary))
            

           
    model = get_model(config.model)
    if is_global_main_process:
        logger.info('Building model...')
        print('Train %d | %s' % (len(train_dataset), val_summary))
        print(f"--- Dataset Split Check: config.dataset.train.split is '{config.dataset.train.split}' ---")
        print('Number of parameters: %d' % count_parameters(model))
        print(f"DeepSpeed will initialize with dataset length: {len(train_dataset)}")

                           
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    it_first = 1
    loaded_scheduler_state = None
    loaded_async_scheduler_last_iter = None

                                                                   
                                                                             
                                                                
                                                                             
                                                                          
           
    deepspeed_config = None
    if args.deepspeed_config:
        with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
            deepspeed_config = json.load(f)
        configured_clip = float(getattr(config.train, 'max_grad_norm', 0.0) or 0.0)
        if configured_clip < 0.0:
            raise ValueError(
                f"train.max_grad_norm must be non-negative, got {configured_clip}",
            )
                                                                       
                                      
        deepspeed_config['gradient_clipping'] = configured_clip

                                                                              
                                                                          
                                                                              
                                                           
        micro_batch = int(config.train.batch_size)
        grad_accum = int(deepspeed_config.get('gradient_accumulation_steps', 1))
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if micro_batch <= 0 or grad_accum <= 0 or world_size <= 0:
            raise ValueError(
                'Invalid distributed batch configuration: '
                f'micro_batch={micro_batch}, grad_accum={grad_accum}, '
                f'world_size={world_size}',
            )
        deepspeed_config.pop('per_device_train_batch_size', None)
        deepspeed_config['train_micro_batch_size_per_gpu'] = micro_batch
        deepspeed_config['train_batch_size'] = micro_batch * grad_accum * world_size

        if is_global_main_process:
            logger.info(
                'DeepSpeed gradient_clipping=%s (from train.max_grad_norm)',
                configured_clip,
            )
            logger.info(
                'DeepSpeed batch contract: micro_batch_per_gpu=%d, '
                'gradient_accumulation_steps=%d, world_size=%d, global_batch=%d',
                micro_batch,
                grad_accum,
                world_size,
                deepspeed_config['train_batch_size'],
            )
                                                                              
                                                                            
                                                                              
                                                        
        args.deepspeed_config = None

    engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        training_data=None,
        collate_fn=None,
        lr_scheduler=None,
        config_params=deepspeed_config,
    )
                                                          
    model = engine
    lr_scheduler = scheduler
    lr_scheduler.optimizer = optimizer
    training_dataloader = engine.deepspeed_io(
        train_dataset,
        batch_size=config.train.batch_size,
        route='train',
        collate_fn=PaddingCollate(),
        num_local_io_workers=args.num_workers,
    )

                                                                            
                                                                               
                                                                           
                                                   
    seed_all(int(config.train.seed) + global_rank)
    train_iterator = inf_iterator(training_dataloader)
                                                          
                                  
                                                          
    world_size = dist.get_world_size()
    per_device_val_batch_size = 5
    if is_global_main_process:
        print(f'World Size: {world_size} | Per Device Batch Size for Val: {per_device_val_batch_size}')

    val_loaders = {}
    for view_name, view_dataset in val_datasets.items():
                                                                             
                                                                               
                                                                              
                                                                              
                                                                      
        val_loaders[view_name] = DataLoader(
            view_dataset,
            batch_size=per_device_val_batch_size,
            collate_fn=PaddingCollate(),
            shuffle=False,
            num_workers=args.num_workers,
        )

    def _unwrap_model(engine_or_model):
        return engine_or_model.module if hasattr(engine_or_model, "module") else engine_or_model

    def _summarize_batch_for_debug(batch):
        origin = batch.get('origin', 'NA')
        if isinstance(origin, (list, tuple)):
            origin_str = ','.join(map(str, origin[:4]))
            if len(origin) > 4:
                origin_str += ',...'
        else:
            origin_str = str(origin)
        generate_flag = batch.get('generate_flag', None)
        if torch.is_tensor(generate_flag):
            generate_count = generate_flag.detach().long().sum(dim=-1).cpu().tolist()
        else:
            generate_count = 'NA'
        aa = batch.get('aa', None)
        length = tuple(aa.shape) if torch.is_tensor(aa) else 'NA'
        return origin_str, generate_count, length

    def _set_flow_debug_context(batch, it):
        base_model = _unwrap_model(model)
        diffusion = getattr(base_model, 'diffusion', None)
        if diffusion is None:
            return
        origin, generate_count, length = _summarize_batch_for_debug(batch)
        diffusion.debug_context = {
            'rank': dist.get_rank() if dist.is_initialized() else 0,
            'iter': it,
            'origin': origin,
            'generate_count': generate_count,
            'length': length,
        }

    def _all_rank_nonfinite_guard(loss_dict, loss, batch, it):
        local_bad_names = []
        for key, value in loss_dict.items():
            if torch.is_tensor(value) and value.dtype.is_floating_point and not torch.isfinite(value).all():
                local_bad_names.append(key)
        if torch.is_tensor(loss) and not torch.isfinite(loss).all() and 'overall' not in local_bad_names:
            local_bad_names.append('overall')

        local_bad = torch.tensor(
            [1 if local_bad_names else 0],
            device=model.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        any_bad = bool(local_bad.item())
        if not any_bad:
            return

        origin, generate_count, length = _summarize_batch_for_debug(batch)
        rank = dist.get_rank() if dist.is_initialized() else 0
        printable_losses = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                printable_losses[key] = value.detach().float().cpu().tolist()
            else:
                printable_losses[key] = value
        print(
            f"[nonfinite-guard] rank={rank} iter={it} local_bad={local_bad_names} "
            f"origin={origin} generate_count={generate_count} length={length} losses={printable_losses}",
            flush=True,
        )
        if dist.is_initialized():
            dist.barrier()
        raise FloatingPointError(f"Non-finite loss detected at iter {it}; all ranks exiting cleanly.")

    def _all_rank_nonfinite_gradient_guard(batch, it):
        local_bad_names = []
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is not None and not torch.isfinite(gradient).all():
                local_bad_names.append(name)

        local_bad = torch.tensor(
            [1 if local_bad_names else 0],
            device=model.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        if not bool(local_bad.item()):
            return

        origin, generate_count, length = _summarize_batch_for_debug(batch)
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            f"[nonfinite-gradient-guard] rank={rank} iter={it} "
            f"local_bad={local_bad_names[:32]} origin={origin} "
            f"generate_count={generate_count} length={length}",
            flush=True,
        )
        if dist.is_initialized():
            dist.barrier()
        raise FloatingPointError(
            f"Non-finite gradient detected at iter {it}; optimizer step was not applied."
        )

    def _choose_async_val_gpu():
        if args.val_async_gpu:
            return args.val_async_gpu
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        devices = [x.strip() for x in visible.split(',') if x.strip()]
        if devices:
            return devices[-1]
        return '0'

    async_val_proc = None

    def _proc_state(pid):
        try:
            with open(f'/proc/{pid}/stat', 'r') as f:
                parts = f.read().split()
            return parts[2] if len(parts) > 2 else None
        except Exception:
            return None

    def launch_async_validation(it):
        global async_val_proc
        if not is_global_main_process:
            return
        if args.debug:
            logger.info(f'[async-val] debug mode: skip iter {it}')
            return
        project_dir = os.path.dirname(os.path.abspath(__file__))
        validate_script = os.path.join(project_dir, 'validate_async.py')
        if not os.path.exists(validate_script):
            logger.warning(f'[async-val] validate_async.py not found: {validate_script}')
            return
        ckpt_tag_dir = os.path.join(ckpt_dir, str(it))
        config_for_val = config_path if os.path.exists(config_path) else args.config
        val_dir = os.path.join(log_dir, 'val_async')
        os.makedirs(val_dir, exist_ok=True)
        pid_path = os.path.join(val_dir, 'async_val.pid')
        if async_val_proc is not None:
            return_code = async_val_proc.poll()
            if return_code is None:
                logger.warning(f'[async-val] previous validation still running pid={async_val_proc.pid}; skip iter {it}')
                return
            logger.info(f'[async-val] previous validation pid={async_val_proc.pid} exited with returncode={return_code}')
            async_val_proc = None
        if os.path.exists(pid_path):
            try:
                with open(pid_path, 'r') as f:
                    old_pid = int(f.read().strip())
                old_state = _proc_state(old_pid)
                if old_state is not None and old_state != 'Z':
                    logger.warning(f'[async-val] previous validation still running pid={old_pid}; skip iter {it}')
                    return
                try:
                    os.remove(pid_path)
                except OSError:
                    pass
            except Exception:
                pass
        val_gpu = _choose_async_val_gpu()
        log_path = os.path.join(val_dir, f'val_{it}.log')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = val_gpu
        env['PYTHONPATH'] = project_dir + os.pathsep + env.get('PYTHONPATH', '')
        cmd = [
            sys.executable,
            validate_script,
            '--checkpoint_dir', ckpt_tag_dir,
            '--config', config_for_val,
            '--output_dir', val_dir,
            '--batch_size', str(args.val_async_batch_size),
            '--num_workers', str(args.val_async_num_workers),
        ]
        with open(log_path, 'ab') as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=project_dir,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        async_val_proc = proc
        with open(pid_path, 'w') as f:
            f.write(str(proc.pid))
        logger.info(f'[async-val] launched iter {it} pid={proc.pid} on CUDA_VISIBLE_DEVICES={val_gpu}; log={log_path}')

    async_scheduler_last_iter = 0

    def _read_latest_async_val_result():
        val_dir = os.path.join(log_dir, 'val_async')
        if not os.path.isdir(val_dir):
            return 0, 0.0

        best_iter = 0
        best_loss = 0.0
        for name in os.listdir(val_dir):
            if not (name.startswith('val_') and name.endswith('.json')):
                continue
            path = os.path.join(val_dir, name)
            try:
                with open(path, 'r') as f:
                    result = json.load(f)
                result_iter = int(result.get('iteration', name[4:-5]))
                result_loss = float(result.get('avg_core_loss', result['avg_loss']))
            except Exception as exc:
                logger.warning(f'[async-val] ignoring unreadable result {path}: {exc}')
                continue
            if result_iter > async_scheduler_last_iter and result_iter > best_iter:
                best_iter = result_iter
                best_loss = result_loss
        return best_iter, best_loss

    def step_async_plateau_scheduler_if_ready():
        global async_scheduler_last_iter
        if not args.val_async or not isinstance(lr_scheduler, ReduceLROnPlateau):
            return

        result_iter = 0
        result_loss = 0.0
        if is_global_main_process:
            result_iter, result_loss = _read_latest_async_val_result()

        payload = torch.tensor([float(result_iter), float(result_loss)], device=model.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.broadcast(payload, src=0)

        result_iter = int(payload[0].item())
        result_loss = float(payload[1].item())
        if result_iter <= async_scheduler_last_iter:
            return

        lr_scheduler.step(result_loss)
        async_scheduler_last_iter = result_iter
        if is_global_main_process:
            logger.info(
                f'[async-val] stepped ReduceLROnPlateau from val_{result_iter}.json '
                f'avg_loss={result_loss:.6f}; lr={optimizer.param_groups[0]["lr"]}'
            )

                                                               
                                                                          
                                                                                    
    checkpoint_arg = args.resume if args.resume is not None else args.finetune
    if checkpoint_arg is not None:
        checkpoint_mode = 'resume' if args.resume is not None else 'finetune'
        ckpt_path = os.path.abspath(checkpoint_arg)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f'{checkpoint_mode} checkpoint does not exist: {ckpt_path}'
            )
        if os.path.isdir(ckpt_path):
            load_dir_base = os.path.dirname(ckpt_path)
            ckp_name = os.path.basename(ckpt_path)
        else:
                                                                         
            tag_dir = os.path.dirname(ckpt_path)
            load_dir_base = os.path.dirname(tag_dir)
            ckp_name = os.path.basename(tag_dir)
        if not load_dir_base or not ckp_name:
            raise ValueError(
                f'Could not parse DeepSpeed checkpoint base/tag from: {ckpt_path}'
            )

        if is_global_main_process:
            logger.info(
                '%s checkpoint: base=%s tag=%s',
                checkpoint_mode.capitalize(),
                load_dir_base,
                ckp_name,
            )
        if checkpoint_mode == 'resume':
            load_tag_out, client_state = engine.load_checkpoint(
                load_dir_base,
                tag=ckp_name,
                load_optimizer_states=True,
                load_lr_scheduler_states=False,
            )
            if load_tag_out is None or not client_state or 'iteration' not in client_state:
                raise RuntimeError(
                    f'DeepSpeed resume did not return iteration state for: {ckpt_path}'
                )
            it_first = int(client_state['iteration']) + 1
            loaded_scheduler_state = client_state.get('lr_scheduler')
            loaded_async_scheduler_last_iter = client_state.get(
                'async_scheduler_last_iter'
            )
        else:
            load_tag_out, _ = engine.load_checkpoint(
                load_dir_base,
                tag=ckp_name,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_only=True,
            )
            if load_tag_out is None:
                raise RuntimeError(
                    f'DeepSpeed finetune checkpoint load failed: {ckpt_path}'
                )
            if is_global_main_process:
                logger.info('Finetune mode: optimizer, scheduler, and iteration reset.')
                                                                              
    if loaded_scheduler_state is not None and hasattr(lr_scheduler, 'load_state_dict'):
        try:
            lr_scheduler.load_state_dict(loaded_scheduler_state)
        except Exception as exc:
            if is_global_main_process:
                logger.warning(f'Could not restore LR scheduler state; continuing fresh: {exc}')
    async_scheduler_last_iter = max(0, it_first - 1)
    if loaded_async_scheduler_last_iter is not None:
        async_scheduler_last_iter = int(loaded_async_scheduler_last_iter)

                                        
    def aggr_ranks_loss(loss_dict):
                            
        reduced_loss_dict = {}
        world_size = dist.get_world_size()
        for key, loss_tensor in loss_dict.items():
                                      
            reduced_tensor = loss_tensor.clone().detach().to(model.device)
                                        
                                        
            dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

                                                           
            reduced_tensor /= world_size
            reduced_loss_dict[key] = reduced_tensor
        loss_dict = reduced_loss_dict                  
        return loss_dict

                                                          
                    
                                                          
    def train(it):
        time_start = current_milli_time()
        model.train()

                                     
        if is_global_main_process and (it <= 5 or it % 1000 == 0):
            print(f"\n[LR Debug] Iter {it}")
            print(f"  optimizer.param_groups[0]['lr'] = {optimizer.param_groups[0]['lr']}")
            print(f"  optimizer.param_groups has 'initial_lr': {'initial_lr' in optimizer.param_groups[0]}")
            print(f"  lr_scheduler type: {type(lr_scheduler).__name__}")
            if hasattr(lr_scheduler, 'base_lrs'):
                print(f"  lr_scheduler.base_lrs: {lr_scheduler.base_lrs}")
            if hasattr(lr_scheduler, 'last_epoch'):
                print(f"  lr_scheduler.last_epoch: {lr_scheduler.last_epoch}")
                                           

        model_dtype = next(model.parameters()).dtype
        batch = recursive_to(next(train_iterator), device=model.device, dtype=model_dtype)
        _set_flow_debug_context(batch, it)

                 
        loss_dict = model(batch)
                                                                          
                                                                               
                                                                         
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        _all_rank_nonfinite_guard(loss_dict, loss, batch, it)

        time_forward_end = current_milli_time()

        if is_global_main_process and it <= 3:
                                                                            
                                                                                
            group_summary = [
                {
                    'index': i,
                    'lr': pg.get('lr', 'N/A'),
                    'weight_decay': pg.get('weight_decay', 'N/A'),
                    'num_parameters': sum(p.numel() for p in pg.get('params', [])),
                }
                for i, pg in enumerate(getattr(optimizer, 'param_groups', []))
            ]
            print(f"[Iter {it}] optimizer={type(optimizer).__name__} groups={group_summary}")
            print(f"[Iter {it}] lr_scheduler={type(lr_scheduler).__name__}")
                                    
        model.backward(loss)
        _all_rank_nonfinite_gradient_guard(batch, it)

                                              
        if is_global_main_process and (it <= 5 or it % 100 == 0):
            total_norm = 0.0
            gradient_groups = {
                'seq_head': ('eps_seq_net',),
                'pos_head': ('eps_crd_net',),
                'rot_head': ('eps_rot_net',),
                'hierarchy': (
                    'rus_role_adapter.chain_encoder',
                    'rus_role_adapter.region_encoder',
                    'rus_role_adapter.complex_encoder',
                    'rus_role_adapter.pair_',
                    'rus_role_adapter.entity_',
                ),
                'rus': (
                    'rus_role_adapter.rus_',
                    'rus_role_adapter.constraint_readers',
                ),
                'identity_game': (
                    'rus_role_adapter.self_query',
                    'rus_role_adapter.role_prototypes',
                    'rus_role_adapter.shared_role_mod',
                    'rus_role_adapter.unique_role_mod',
                    'rus_role_adapter.game_context',
                    'rus_role_adapter.payoff_head',
                    'rus_role_adapter.role_',
                ),
                'directed_modal': ('identity_directed_modal',),
                'coupling_adapters': ('game_adapters',),
                'semantic_clock': ('semantic_clock_log_gamma',),
            }
            group_squared_norm = {name: 0.0 for name in gradient_groups}
            clock_value = None
            clock_grad_abs = None
            for name, param in model.named_parameters():
                if 'semantic_clock_log_gamma' in name:
                    clock_value = param.detach().float().exp().item()
                    if param.grad is not None:
                        clock_grad_abs = param.grad.detach().float().abs().mean().item()
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    for group, patterns in gradient_groups.items():
                        if any(pattern in name for pattern in patterns):
                            group_squared_norm[group] += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            group_summary = ' '.join(
                f'{name}={value ** 0.5:.6f}'
                for name, value in group_squared_norm.items()
            )
            logger.info(
                f'[grad-groups] Iter {it:05d} | total={total_norm:.6f} | {group_summary}'
            )
            if clock_value is not None:
                writer.add_scalar('train/semantic_clock_gamma', clock_value, it)
                if clock_grad_abs is not None:
                    writer.add_scalar('train/semantic_clock_log_gamma_grad_abs', clock_grad_abs, it)
                logger.info(
                    f'[semantic-clock] Iter {it:05d} | gamma={clock_value:.8f} '
                    f'| grad_abs={clock_grad_abs if clock_grad_abs is not None else float("nan"):.8e}'
                )

        model.step()

        time_backward_end = current_milli_time()

                                             
                            
        if dist.is_initialized():
            loss_dict = aggr_ranks_loss(loss_dict)

        if is_global_main_process:
            log_losses(loss_dict, it, 'train', logger, writer, others={
                                         
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })

                                                          
                       
                                                          
    def validate(it):
        model.eval()

        model_dtype = next(model.parameters()).dtype
        val_t_steps = list(getattr(config.train, 'val_t_steps', [10, 30, 50, 70, 90]))
        rank = dist.get_rank() if dist.is_initialized() else 0
        view_avg = {}
        for view_index, (view_name, val_loader) in enumerate(val_loaders.items()):
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
                for i, batch in enumerate(
                        tqdm(
                            val_loader,
                            desc=f'Validate({view_name})',
                            dynamic_ncols=True,
                            disable=not is_global_main_process,
                        )):
                    batch = recursive_to(batch, device=model.device, dtype=model_dtype)
                    batch_size = batch['aa'].shape[0]

                    for t_step in val_t_steps:
                        seed_all(
                            int(config.train.seed)
                            + 100000
                            + view_index * 1000000
                            + i * 100
                            + int(t_step)
                        )
                        fixed_t = torch.full(
                            (batch_size,),
                            fill_value=int(t_step),
                            dtype=torch.long,
                            device=model.device,
                        )
                        loss_dict = model(batch, t=fixed_t)
                        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                        loss_dict = dict(loss_dict)
                        loss_dict['overall'] = loss
                        loss_dict['core'] = compute_core_loss(loss_dict)
                                                                              
                                                                    
                        loss_tape.update(loss_dict, batch_size)
                        for key, values in diagnostics_by_t[int(t_step)].items():
                            if key in loss_dict:
                                values.append(loss_dict[key].item())

            avg_loss_dict, _ = loss_tape.compute_avg()
            view_avg[view_name] = avg_loss_dict
            if is_global_main_process:
                loss_tape.log(it, logger, writer, f'val/{view_name}')
                for key in next(iter(diagnostics_by_t.values())):
                    summary = ' '.join(
                        f't{t_step}={sum(metrics[key]) / len(metrics[key]):.4f}'
                        for t_step, metrics in diagnostics_by_t.items()
                        if metrics[key]
                    )
                    logger.info(
                        f'[val-by-t/{view_name}/{key}] Iter {it:05d} | {summary}'
                    )

        avg_loss = torch.stack([view['overall'] for view in view_avg.values()]).mean().item()
        avg_core_loss = torch.stack([view['core'] for view in view_avg.values()]).mean().item()
        if is_global_main_process:
            logger.info(
                f'[val-combined] Iter {it:05d} | loss {avg_loss:.4f} | '
                f'core {avg_core_loss:.4f} | monitor={"+".join(view_avg)}'
            )
            writer.add_scalar('val/loss_combined', avg_loss, it)
            writer.add_scalar('val/loss_core_monitor', avg_core_loss, it)
            writer.flush()

                                                                               
                                                                      
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(avg_core_loss)
        else:
            lr_scheduler.step()

                                                                          
                                                                            
                                                                      
        seed_all(int(config.train.seed) + global_rank + int(it) * 10000019)
        return avg_loss, avg_core_loss

                                                          
               
                                                          
    try:
             
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if args.val_async and isinstance(lr_scheduler, ReduceLROnPlateau) and (it <= 5 or it % 10 == 0):
                step_async_plateau_scheduler_if_ready()
                   
            if it % config.train.val_freq == 0:
                if args.val_async:
                    avg_val_loss = None
                else:
                    avg_val_loss, avg_val_core_loss = validate(it)
                    print(f'avg_val_loss {avg_val_loss}, avg_val_core_loss {avg_val_core_loss}, CUDA: {dist.get_rank()}')

                                                                                                  
                engine.save_checkpoint(
                    ckpt_dir,                                                  
                    tag=str(it),                                             
                    client_state={
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                        'avg_val_core_loss': None if args.val_async else avg_val_core_loss,
                        'lr_scheduler': (
                            lr_scheduler.state_dict()
                            if hasattr(lr_scheduler, 'state_dict') else None
                        ),
                        'async_scheduler_last_iter': async_scheduler_last_iter,
                    }
                )

                if args.val_async:
                    launch_async_validation(it)
                    if not isinstance(lr_scheduler, ReduceLROnPlateau):
                        lr_scheduler.step()

                                                       
                if is_global_main_process and not args.debug:
                    print(f'---------------it {it}-----avg_val_loss {avg_val_loss}-----saved checkpoint {it}...........')
                    logger.info(f"Saved checkpoint (DS native) for iteration {it} to {os.path.join(ckpt_dir, str(it))}")

    except KeyboardInterrupt:
        logger.info('Terminating...')

