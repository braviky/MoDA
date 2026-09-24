import torch
import torch.nn as nn

from moda.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from moda.modules.common.layers import AngularEncoding
from moda.utils.protein.constants import BBHeavyAtom, AA


                                                 
class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms      
        self.max_aa_types = max_aa_types      
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)            
        self.dihed_embed = AngularEncoding()
        self.type_embed = nn.Embedding(10, feat_dim, padding_idx=0)                               
        infeat_dim = feat_dim + (self.max_aa_types*max_num_atoms*3) + self.dihed_embed.get_out_dim(3) + feat_dim                              
        self.mlp = nn.Sequential(                 
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),                   
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),            
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),            
            nn.Linear(feat_dim, feat_dim)            
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type, structure_mask=None, sequence_mask=None):
        
        N, L = aa.size()                                              
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]                             

                            
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

                                           
        if sequence_mask is not None:                                   
                                                 
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa)                                                      

                                  
        R = construct_3d_basis(                                       
            pos_atoms[:, :, BBHeavyAtom.CA],                        
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)                                                                                                 
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))                      

        aa_expand  = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types*self.max_num_atoms*3)                                                                    
        if structure_mask is not None:
                                                 
            crd_feat = crd_feat * structure_mask[:, :, None]               

                                          
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]                                                          
        dihed_feat = dihed_feat.reshape(N, L, -1)                      
        if structure_mask is not None:
                                                 
            dihed_mask = torch.logical_and(                                                                                   
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1), 
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )                                                                     
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

                                         
        type_feat = self.type_embed(fragment_type)               
                                                                                                   
                                        
                                                                                    
                                                                                  
                                                                         
        encoder_input = torch.cat([aa_feat, crd_feat, dihed_feat, type_feat], dim=-1)
        encoder_input = encoder_input.to(dtype=self.mlp[0].weight.dtype)
        out_feat = self.mlp(encoder_input)            
        out_feat = out_feat * mask_residue[:, :, None]              
        return out_feat
