import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants

                                               
                                                                         
                                  
                                                      
def get_epitope_mask(antigen_mask, gen_mask, pos_heavyatom, mask_heavyatom,
                         mode: str = "ca", cutoff: float = 4.0):
    
    assert mode in ["ca", "all_atom", "ca_then_all_atom"], f"Unknown mode {mode}"

    L, n_atom, _ = pos_heavyatom.shape             

    if gen_mask.sum() == 0:
        return torch.zeros(L, dtype=torch.bool), torch.tensor([], dtype=torch.long), torch.empty(0, n_atom, 3)

                             
    pos_alpha = pos_heavyatom[:, constants.BBHeavyAtom.CA]          
    gen_ca = pos_alpha[gen_mask]                    
    ag_ca = pos_alpha[antigen_mask]                
    dist_ca = torch.cdist(gen_ca, ag_ca)                 
    ag_min_dist_ca = dist_ca.min(dim=0)[0]           

    if mode == "ca":
        epitope_mask_local = (ag_min_dist_ca <= cutoff)
        epitope_indices = antigen_mask.nonzero(as_tuple=True)[0][epitope_mask_local]
        epitope_coords = pos_heavyatom[epitope_indices]
        epitope_mask = torch.zeros(L, dtype=torch.bool)
        epitope_mask[epitope_indices] = True
        return epitope_mask, epitope_indices, epitope_coords

                               
    gen_atoms = pos_heavyatom[gen_mask]                              
    ag_atoms = pos_heavyatom[antigen_mask]                          
    gen_atom_mask = mask_heavyatom[gen_mask]                      
    ag_atom_mask = mask_heavyatom[antigen_mask]                  

        
    gen_atoms_flat = gen_atoms.reshape(-1, 3)                     
    ag_atoms_flat = ag_atoms.reshape(-1, 3)                      
    dist_all = torch.cdist(gen_atoms_flat, ag_atoms_flat)                               

               
    gen_mask_flat = gen_atom_mask.reshape(-1)
    ag_mask_flat = ag_atom_mask.reshape(-1)
    valid_mask = gen_mask_flat[:, None] & ag_mask_flat[None, :]
    dist_all = torch.where(valid_mask, dist_all, torch.full_like(dist_all, float("inf")))

                                             
    dist_all = dist_all.view(gen_atoms.shape[0], gen_atoms.shape[1],
                             ag_atoms.shape[0], ag_atoms.shape[1])

                         
    min_dist_all = dist_all.min(dim=1)[0].min(dim=0)[0].min(dim=1)[0]

    if mode == "all_atom":
        epitope_mask_local = (min_dist_all <= cutoff)
    elif mode == "ca_then_all_atom":
                 
        candidate_mask_ag = (ag_min_dist_ca <= 8.0)
        epitope_mask_local = torch.zeros_like(candidate_mask_ag, dtype=torch.bool)
        epitope_mask_local[candidate_mask_ag] = (min_dist_all[candidate_mask_ag] <= cutoff)
    else:
        raise ValueError(f"Unsupported mode {mode}")
    epitope_indices = antigen_mask.nonzero(as_tuple=True)[0][epitope_mask_local]
    epitope_coords = pos_heavyatom[epitope_indices]
    epitope_mask = torch.zeros(L, dtype=torch.bool)
    epitope_mask[epitope_indices] = True

    return epitope_mask, epitope_indices, epitope_coords                                         

                                                                                                                             
@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):                                                                           
                                                                        
                                                                                                                                         
                                
    def __init__(self, initial_patch_size=128, antigen_size=128):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size

                                                                     
    def _center(self, data, origin):
        origin = origin.reshape(1, 1, 3)
        data['pos_heavyatom'] -= origin            
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        data['origin'] = origin.reshape(3)
        return data

    def __call__(self, data):                   
        anchor_flag = data['anchor_flag']                  
                                                                 
                                                  

        pos_heavyatom = data['pos_heavyatom']                          
        anchor_points = pos_heavyatom[anchor_flag, constants.BBHeavyAtom.CA]                                    
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)                       
        antibody_mask = torch.logical_not(antigen_mask)                       

        if anchor_flag.sum().item() == 0:                                
                                                           
            data_patch = _mask_select_data(
                data = data,
                mask = antibody_mask,
            )
            data_patch = self._center(
                data_patch,
                origin = data_patch['pos_heavyatom'][:, constants.BBHeavyAtom.CA].mean(dim=0)
            )
            return data_patch

        pos_alpha = pos_heavyatom[:, constants.BBHeavyAtom.CA]                          
                                                         
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]                                
        initial_patch_idx = torch.topk(
            dist_anchor,
            k = min(self.initial_patch_size, dist_anchor.size(0)),                                                                     
            largest=False,
        )[1]                                                                                                   

                                                            
                                                                                                                                                                                                          
                                                       
        if antigen_mask.sum() > 0:         
            epitope_mask_ag, _, _ = get_epitope_mask(
                antigen_mask=antigen_mask,
                gen_mask=data['generate_flag'],
                pos_heavyatom=pos_heavyatom,
                mask_heavyatom=data['mask_heavyatom'], mode="all_atom",
                cutoff=4.0)
        else:
            epitope_mask_ag = torch.zeros_like(antigen_mask, dtype=torch.bool)

                                                                     
        dist_anchor_antigen = dist_anchor.masked_fill(
            mask = antibody_mask,                          
            value = float('+inf')
        )                                          
        antigen_patch_idx = torch.topk(                                                   
            dist_anchor_antigen, 
            k = min(self.antigen_size, antigen_mask.sum().item()),                                                                      
            largest=False, sorted=True
        )[1]                                                                 

                              
        patch_mask = torch.logical_or(
            data['generate_flag'],
            data['anchor_flag'],
        )         
        patch_mask[initial_patch_idx] = True          
                                    
        patch_mask[antigen_patch_idx] = True          
                                                                          
                                                                                 
                                                                                
                                                                              
                                                                             
                                                                              
                                              

                                                
        data['epitope_mask'] = epitope_mask_ag                   
        data['ag_non_epitope_mask'] = antigen_mask & (~epitope_mask_ag)                   
                                                                                                                                                 
        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]                                                                                                    

        data_patch = _mask_select_data(data, patch_mask)                                              

        data_patch = self._center(                                    
            data_patch,
            origin = anchor_points.mean(dim=0)                              
        )

               
                         
                                                                                     
                                                                                                                                                                    
         
                                                          
                                                                                                              
                                                                                                                          
         
                                                                                               
                                                                  
                                                                                        
                                                                         
         
                                                                     
                                                                                                                                           

        data_patch['patch_idx'] = patch_idx                    
          
                                                
        

        return data_patch
