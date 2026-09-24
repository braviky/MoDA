import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import datetime

from moda.datasets import get_dataset
from moda.datasets.sabdab import *
from moda.models import get_model
from moda.modules.common.geometry import reconstruct_backbone_partially
from moda.modules.common.so3 import so3vec_to_rotation
from moda.utils.inference import RemoveNative
from moda.utils.protein.writers import save_pdb
from moda.utils.train import recursive_to
from moda.utils.misc import *
from moda.utils.data import *
from moda.utils.transforms import *
from moda.utils.inference import *
from design_utils import *

TEST_MODA_PATH = f'./testdataset_moda.txt'
TEST_MODA = [
    '5xku', '7chf', '7chf', '7che', '5tlk',
    '5tlj', '5tlk', '5w9h', '5tlj', '5tl5',
    '7bwj', '7d6i', '8ds5', '5w9h', '7chb',
    '5w9h', '7che', '5tlk', '5tlk'
]
TEST_RABD_PATH = f'./testdataset_RAbD.txt'
TEST_RABD = [
    '4cmh','3nid','5mes','1ic7','1ncb',
    '5bv7','4lvn','4ki5','4etq','2adf',
    '4ydk','2cmr','5b8c','3bn9','2xqy'
    '1a2y','3o2d','1fe8','1n8z','4g6j',
    '3mxw','2b2x','3hi6','1osp','1uj3'
    '3ffd','2ypv','4ffv','4h8w''2vxt',
    '4dvr','3s35','3w9e','5f9o','1iqd',
    '5d96','4g6m','4ot1','5j13','5l6y',
    '3l95','3cx5','2xwt','5nuz','4fqj',
    '3k2u','1w72','3rkd','3h3b','4qci',
    '4dtg','4xnq','5en2','5d93','5ggs',
    '3uzq','2ghw','2dd8','5hi4','1a14',
]

                                                      
                                     
                                    
 
                        
                                     
                                                                                           
                               
                                   
                                                              
                                
                
                                                       
                                                                            
                                    
                                   
                                                       
                                       
                                  
                                                 
                                               
                
                                          
                                                                                           
                               
                                                                   
                            
            
                                                   
                                
                               
                                                     
                                    
                           
                                    
                                   
            
                                 
                               
                             
                            
            
                                                   
                                
                               
                                             
                            
                                    
                                   
            
                                  
                                                                                           
                               
                                   
                                                              
                                
                
                                                       
                                                                            
                                                             
                                        
                                       
                                                                       
                                                       
                                      
                                           
                                                     
                                                   
                    
           
                                                           
                          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)                         
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_multicdrs.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

                  
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

             
    dataset = get_dataset(config.dataset.test)
    print(f'dataset {len(dataset)}')           

    for data_id in range(len(dataset)):
        get_structure = lambda: dataset[data_id]

                 
        structure_ = get_structure()              
        structure_id = structure_['id']
        print(f'struc {structure_id}')

        tag_postfix = '_%s' % args.tag if args.tag else ''
                 
                                         
                                                                                                                                              
        log_dir = get_new_log_dir(os.path.join(args.out_root, config_name + tag_postfix), prefix='%s' % structure_['id'])
        logger = get_logger('sample', log_dir)
        logger.info('Data ID: %s' % structure_['id'])

        data_native = MergeChains()(structure_)
        save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

                                   
        logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
        ckpt = torch.load(config.model.checkpoint, map_location='cpu')
                                                  

                                   
        model = get_model(config.model).to(args.device)
        try:
            lsd = model.load_state_dict(ckpt)
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint is not compatible with the flow-matching model. "
                "Use a checkpoint trained under MoDA_flowmatching_v1, or explicitly migrate encoder weights."
            ) from e
        logger.info('Loaded checkpoint state: %s' % (lsd, ))
        logger.info(str(lsd))
        if hasattr(model, 'assert_runtime_contract'):
            model.assert_runtime_contract()

                            
        heavy_chain = structure_['heavy']['chain_id'][0].strip()             
        light_chain = structure_['light']['chain_id'][0].strip()
        print(f'heavy_chain {heavy_chain};; light chain {light_chain}')

        data_variants = create_data_variants(                                                                                           
            config=config,
            structure_factory=get_structure,              
            heavy_id=heavy_chain,
            light_id=light_chain,
        )

                                 
        metadata = {
            'identifier': structure_id,
                                  
            'index': data_id,             
            'config': args.config,
            'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
        }
        with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        run_eval_dir = os.path.dirname(log_dir)
        for export_idx, export_variant in enumerate(data_variants):
            export_eval_records(
                run_dir=run_eval_dir,
                structure_id=structure_id,
                data_native=data_native,
                variant=export_variant,
                case_dir=log_dir if export_idx == 0 else None,
            )

                        
        collate_fn = PaddingCollate(eight=False)
        inference_tfm = [ PatchAroundAnchor(), ]
        if 'abopt' not in config.mode:                                                
            inference_tfm.append(RemoveNative(
                remove_structure = config.sampling.sample_structure,
                remove_sequence = config.sampling.sample_sequence,
            ))
        inference_tfm = Compose(inference_tfm)
        print(f'{config.mode} totally {len(data_variants)} variants')

        for j, variant in enumerate(data_variants):                                                  
            os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
            logger.info(f"Start sampling for: {variant['tag']}")

            save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))                                

            data_cropped = inference_tfm(                                    
                copy.deepcopy(variant['data'])
            )

            data_list_repeat = [ data_cropped ] * config.sampling.num_samples                                
            loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)                           

            count = 0
            for batch_index, batch in enumerate(tqdm(loader, desc=variant['name'], dynamic_ncols=True)):                    
                torch.set_grad_enabled(False)
                model.eval()
                batch = recursive_to(batch, args.device)
                if 'abopt' in config.mode:
                                                                
                    traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                        'pbar': True,
                        'sample_structure': config.sampling.sample_structure,
                        'sample_sequence': config.sampling.sample_sequence,
                    })
                else:
                                    
                                                                                                                                     
                    sample_opt = {
                        'pbar': True,
                        'sample_structure': config.sampling.sample_structure,          
                        'sample_sequence': config.sampling.sample_sequence,          
                    }
                    async_mode = config.sampling.get('async_mode', None)
                    if async_mode is not None:
                        sample_opt['async_mode'] = async_mode
                    traj_batch = model.sample(batch, sample_opt=sample_opt)

                                                                                                      
                aa_new = traj_batch[0][2]                                          
                pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                    pos_ctx = batch['pos_heavyatom'],
                    R_new = so3vec_to_rotation(traj_batch[0][0]),
                    t_new = traj_batch[0][1],
                    aa = aa_new,
                    chain_nb = batch['chain_nb'],
                    res_nb = batch['res_nb'],
                    mask_atoms = batch['mask_heavyatom'],
                    mask_recons = batch['generate_flag'],
                )
                aa_new = aa_new.cpu()
                pos_atom_new = pos_atom_new.cpu()
                mask_atom_new = mask_atom_new.cpu()

                for i in range(aa_new.size(0)):
                    data_tmpl = variant['data']
                    aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                    mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                    pos_ha  = (
                        apply_patch_to_tensor(
                            data_tmpl['pos_heavyatom'],
                            pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(),
                            data_cropped['patch_idx']
                        )
                    )

                    save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                    save_pdb({
                        'chain_nb': data_tmpl['chain_nb'],
                        'chain_id': data_tmpl['chain_id'],
                        'resseq': data_tmpl['resseq'],
                        'icode': data_tmpl['icode'],
                                   
                        'aa': aa,
                        'mask_heavyatom': mask_ha,
                        'pos_heavyatom': pos_ha,
                    }, path=save_path)
                                
                                                               
                                                               
                                                           
                                                         
                                     
                                          
                                                             
                                                                                                    
                                                                                                  
                    count += 1

            logger.info(f'For {data_id} in dataset ({len(dataset)}), finished variant {j} / {len(variant)}.\n')


if __name__ == '__main__':
    main()
