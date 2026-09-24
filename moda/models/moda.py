import hashlib
import json

import torch
import torch.nn as nn

from moda.modules.common.geometry import construct_3d_basis
from moda.modules.common.so3 import rotation_to_so3vec
from moda.modules.encoders.residue import ResidueEmbedding
from moda.modules.encoders.pair import PairEmbedding
from moda.modules.flow_full import FullDPM
from moda.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


def _plain_config(value):
    if isinstance(value, dict):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _model_config_digest(cfg):
    payload = _plain_config(cfg)
                                                                             
    payload.pop('checkpoint', None)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


@register_model('moda')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        runtime_digest = _model_config_digest(cfg)
        self.register_buffer('_model_config_contract', runtime_digest.clone(), persistent=True)
        self.register_buffer('_runtime_model_config_digest', runtime_digest, persistent=False)
                
        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]                     
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim,
                                              num_atoms)                                                
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim,
                                        num_atoms)                                                      

        self.diffusion = FullDPM(
            cfg.res_feat_dim,       
            cfg.pair_feat_dim,      
            **cfg.diffusion,
        )
        self.last_sample_analysis = []
        self.last_sample_analysis_summary = {'enabled': False, 'steps': 0}

    def assert_runtime_contract(self):
        if not torch.equal(
            self._model_config_contract.detach().cpu(),
            self._runtime_model_config_digest.detach().cpu(),
        ):
            raise RuntimeError(
                'Runtime model configuration differs from the checkpoint model contract. '
                'Build validation/inference from the model block saved with training; only '
                'the checkpoint path may differ.'
            )
         
                                                                         
             
                  
                                              
                                                  
             
                                                                 
                                    
                                                                                        
                                                            
                                                                        
           
     
                                                                        
                                                                   
                                           
                                        
                                                                                                                                                                                                  
                             
                                     
                                         
                                               
                                                 
                                                   
                                            
                                          
                                                                                                                                         
     
                                      
                                                                                                  
                             
                                     
                                         
                                               
                                                 
                                            
                                          
                                                                                                                                            
     
                                               
                                                           
                                                          
                                                          
           
                                                          
     
                                                                                                                                                          
    def encode(self, batch, remove_structure, remove_sequence):        
        
                                                               
                                  
        context_mask = torch.logical_and(                                             
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_flag']                                   
        )

        structure_mask = context_mask if remove_structure else None   
        sequence_mask = context_mask if remove_sequence else None
                                         
        res_feat = self.residue_embed(
                                                                                                                                                                                                
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            fragment_type=batch['fragment_type'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )                                                                                                                              

        pair_feat = self.pair_embed(
                                                                                                
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )                                                                                                                                 

        R = construct_3d_basis(              
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p                                                                                                                


    def forward(self, batch, t=None):                             
        self.assert_runtime_contract()
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
                                                            
        res_feat, pair_feat, R_0, p_0 = self.encode(
                                                                                                                          
            batch,
            remove_structure=self.cfg.get('train_structure', True),
            remove_sequence=self.cfg.get('train_sequence', True)
        )                                                                                                                
        v_0 = rotation_to_so3vec(R_0)                                    
        s_0 = batch['aa']        
                                               
                                                           
        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
                                                                     
            denoise_structure=self.cfg.get('train_structure', True),
            denoise_sequence=self.cfg.get('train_sequence', True),
            t=t,
            batch=batch,
        )
        return loss_dict

    @torch.no_grad()
    def sample(      
            self,
            batch,
            sample_opt={
                'sample_structure': True,
                'sample_sequence': True,
            }
    ):
        self.assert_runtime_contract()
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(                             
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)      
        s_0 = batch['aa']        
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, batch=batch, **sample_opt)
        self.last_sample_analysis = getattr(self.diffusion, 'last_sample_analysis', [])
        self.last_sample_analysis_summary = getattr(
            self.diffusion, 'last_sample_analysis_summary', {'enabled': False, 'steps': 0}
        )
        self._attach_native_identity_audit(self.last_sample_analysis_summary, traj, batch)
        return traj

    @staticmethod
    def _attach_native_identity_audit(summary, traj, batch):
        
        if not isinstance(summary, dict) or not traj or not isinstance(batch, dict):
            return
        if 'aa' not in batch or 'generate_flag' not in batch or 'mask' not in batch:
            return
        generated = batch['generate_flag'].bool()
        valid = generated & batch['mask'].bool()
        native = batch.get('native_aa')
        if native is None:
                                                                                    
                                                                                     
                                                                      
            native = batch['aa']
            valid = valid & (native >= 0) & (native < 20)
        sampled = traj[0][2].to(native.device)
        match = (sampled == native) & valid
        summary['native_sequence_match'] = float(match.sum().float().div(valid.sum().clamp_min(1)).item())
        summary['native_sequence_match_per_residue'] = match.masked_fill(~valid, False).cpu().tolist()
        cdr = batch.get('cdr', batch.get('cdr_flag'))
        if cdr is not None:
            summary['native_sequence_match_by_cdr'] = [
                float(match[valid & (cdr == cdr_id)].float().mean().item())
                if bool((valid & (cdr == cdr_id)).any()) else None
                for cdr_id in range(1, 7)
            ]

    @torch.no_grad()
    def optimize(
            self,
            batch,
            opt_step,
            optimize_opt={
                'sample_structure': True,
                'sample_sequence': True,
            }
    ):
        self.assert_runtime_contract()
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=optimize_opt.get('sample_structure', True),
            remove_sequence=optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res,
                                       batch=batch,
                                       **optimize_opt)
        self.last_sample_analysis = getattr(self.diffusion, 'last_sample_analysis', [])
        self.last_sample_analysis_summary = getattr(
            self.diffusion, 'last_sample_analysis_summary', {'enabled': False, 'steps': 0}
        )
        self._attach_native_identity_audit(self.last_sample_analysis_summary, traj, batch)
        return traj
