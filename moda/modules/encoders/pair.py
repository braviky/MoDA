import torch
import torch.nn as nn
import torch.nn.functional as F

from moda.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from moda.modules.common.layers import AngularEncoding
from moda.utils.protein.constants import BBHeavyAtom, AA


class PairEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms      
        self.max_aa_types = max_aa_types      
        self.max_relpos = max_relpos           
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)              
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)           

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)             
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(                 
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()                             
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)                    

        infeat_dim = feat_dim+feat_dim+feat_dim+feat_dihed_dim                         
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        
        N, L = aa.size()

                            
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]                                      
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]                                                       
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None                                                                       

                                 
        if sequence_mask is not None:
                                                 
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))                             
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]                                                                                                             
        feat_aapair = self.aa_pair_embed(aa_pair)                               
    
                                                            
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])                                          
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )                                                    
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]                                                                                                        

                                      
        d = angstrom_to_nm(torch.linalg.norm(                                                   
            pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            dim = -1, ord = 2,
        )).reshape(N, L, L, -1)                                   
        c = F.softplus(self.aapair_to_distcoef(aa_pair))                                                                                                
        d_gauss = torch.exp(-1 * c * d**2)                               
        mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, -1)                                                                                                        
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)                                                                                                       
        if pair_structure_mask is not None:
                                                 
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]                                                                        

                            
        dihed = pairwise_dihedrals(pos_atoms)                              
        feat_dihed = self.dihedral_embed(dihed)                               
        if pair_structure_mask is not None:
                                                 
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

             
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)                 
        feat_all = feat_all * mask_pair[:, :, :, None]                                                        

        return feat_all

