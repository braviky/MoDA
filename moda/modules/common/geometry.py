import torch
import torch.nn.functional as F

from moda.utils.protein.constants import (
    BBHeavyAtom, 
    backbone_atom_coordinates_tensor,
    bb_oxygen_coordinate_tensor,
)
from .topology import get_terminus_flag


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def pairwise_distances(x, y=None, return_v=False):
    
    if y is None: y = x
    v = x.unsqueeze(2) - y.unsqueeze(1)                
    d = safe_norm(v, dim=-1)
    if return_v:
        return d, v
    else:
        return d


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):                                       
    
    v1 = p1 - center               
    e1 = normalize_vector(v1, dim=-1)        

    v2 = p2 - center               
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)               

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)                                                                                             
    return mat


def local_to_global(R, t, p):
    
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    compute_dtype = torch.promote_types(R.dtype, p.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    R = R.to(compute_dtype)
    t = t.to(compute_dtype)
    p = p.to(compute_dtype)

    p = p.view(N, L, -1, 3).transpose(-1, -2)                                 
    q = torch.matmul(R, p) + t.unsqueeze(-1)                  
    q = q.transpose(-1, -2).reshape(p_size)                                                     
    return q


def global_to_local(R, t, q):
    
    assert q.size(-1) == 3         
    q_size = q.size()                 
    N, L = q_size[0], q_size[1]

    compute_dtype = torch.promote_types(R.dtype, q.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    R = R.to(compute_dtype)
    t = t.to(compute_dtype)
    q = q.to(compute_dtype)

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)                                             
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))                                                                                                                                   
    p = p.transpose(-1, -2).reshape(q_size)                                                                                 
    return p


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def compose_rotation_and_translation(R1, t1, R2, t2):
    
    compute_dtype = torch.promote_types(
        torch.promote_types(R1.dtype, R2.dtype),
        torch.promote_types(t1.dtype, t2.dtype),
    )
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    R1, t1, R2, t2 = (
        R1.to(compute_dtype),
        t1.to(compute_dtype),
        R2.to(compute_dtype),
        t2.to(compute_dtype),
    )
    R_new = torch.matmul(R1, R2)                  
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


                                                    
                      
 
                                                                       
                                                         
def quaternion_to_rotation_matrix(quaternions):
    
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


                                                    
                      
 
                                                                       
                                                         
"""
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
def quaternion_1ijk_to_rotation_matrix(q):
    
    b, c, d = torch.unbind(q, dim=-1)
    sq_norm = b ** 2 + c ** 2 + d ** 2
    s = torch.sqrt(1 + sq_norm + 1e-12)
    a, b, c, d = 1/s, b/s, c/s, d/s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2,  2*b*c - 2*a*d,  2*b*d + 2*a*c,
            2*b*c + 2*a*d,  a**2 - b**2 + c**2 - d**2,  2*c*d - 2*a*b,
            2*b*d - 2*a*c,  2*c*d + 2*a*b,  a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def repr_6d_to_rotation_matrix(x):
    
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    mat = torch.cat([
        b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)
    ], dim=-1)                      
    return mat


def dihedral_from_four_points(p0, p1, p2, p3):
    
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999) )
    dihed = torch.nan_to_num(dihed)
    return dihed


def knn_gather(idx, value):
    
    N, d = idx.size(1), value.size(-1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, d)                    
    value = value.unsqueeze(1).repeat(1, N, 1, 1)                 
    return torch.gather(value, dim=2, index=idx)


def knn_points(q, p, K):
    
    _, L, _ = p.size()
    d = pairwise_distances(q, p)             
    dist, idx = d.topk(min(L, K), dim=-1, largest=False)                        
    return dist, idx, knn_gather(idx, p)


def angstrom_to_nm(x):                   
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):                                    
    
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]              
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)                                       
    omega_mask = torch.logical_not(N_term_flag)                         
    phi_mask = torch.logical_not(N_term_flag)                        
    psi_mask = torch.logical_not(C_term_flag)                        

                                        
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), 
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

                              
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed                                               
    return bb_dihedral, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    
    N, L = pos_atoms.shape[:2]
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]              
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3), 
        pos_CA[:,None,:].expand(N, L, L, 3), 
        pos_C[:,None,:].expand(N, L, L, 3)
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:,:,None].expand(N, L, L, 3), 
        pos_CA[:,:,None].expand(N, L, L, 3), 
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3)
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed


def apply_rotation_matrix_to_rot6d(R, O):
    
    u1, u2 = O[..., :3, None], O[..., 3:, None]              
    v1 = torch.matmul(R, u1).squeeze(-1)              
    v2 = torch.matmul(R, u2).squeeze(-1)
    return torch.cat([v1, v2], dim=-1)


def normalize_rot6d(O):
    
    u1, u2 = O[..., :3], O[..., 3:]               
    v1 = F.normalize(u1, p=2, dim=-1)             
    v2 = F.normalize(u2 - project_v2v(u2, v1), p=2, dim=-1)
    return torch.cat([v1, v2], dim=-1)
    

def reconstruct_backbone(R, t, aa, chain_nb, res_nb, mask):
    
    N, L = aa.size()
                                                                                       
    bb_coords = backbone_atom_coordinates_tensor.clone().to(t)              
    oxygen_coord = bb_oxygen_coordinate_tensor.clone().to(t)             
    aa = aa.clamp(min=0, max=20)                

    bb_coords = bb_coords[aa.flatten()].reshape(N, L, -1, 3)                  
    oxygen_coord = oxygen_coord[aa.flatten()].reshape(N, L, -1)             
    bb_pos = local_to_global(R, t, bb_coords)                                                  

                       
    bb_dihedral, _ = get_backbone_dihedral_angles(bb_pos, chain_nb, res_nb, mask)
    psi = bb_dihedral[..., 2]           
                                  
    sin_psi = torch.sin(psi).reshape(N, L, 1, 1)
    cos_psi = torch.cos(psi).reshape(N, L, 1, 1)
    zero = torch.zeros_like(sin_psi)
    one = torch.ones_like(sin_psi)
    row1 = torch.cat([one, zero, zero], dim=-1)                   
    row2 = torch.cat([zero, cos_psi, -sin_psi], dim=-1)               
    row3 = torch.cat([zero, sin_psi, cos_psi], dim=-1)                
    R_psi = torch.cat([row1, row2, row3], dim=-2)                     

                                                                        
    R_psi, t_psi = compose_chain([
        (R, t),           
        (R_psi, torch.zeros_like(t)),                  
    ])
    O_pos = local_to_global(R_psi, t_psi, oxygen_coord.reshape(N, L, 1, 3))

    bb_pos = torch.cat([bb_pos, O_pos], dim=2)                
    return bb_pos
    

def reconstruct_backbone_partially(pos_ctx, R_new, t_new, aa, chain_nb, res_nb, mask_atoms, mask_recons):                                                  
    
    N, L, A = mask_atoms.size()

    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]                   
    pos_recons = reconstruct_backbone(R_new, t_new, aa, chain_nb, res_nb, mask_res)                        
    pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A-4), value=0)                                                          

    pos_new = torch.where(                                            
        mask_recons[:, :, None, None].expand_as(pos_ctx),
        pos_recons, pos_ctx
    )                 
                                        
    mask_bb_atoms = torch.zeros_like(mask_atoms)
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(
        mask_recons[:, :, None].expand_as(mask_atoms),
        mask_bb_atoms, mask_atoms
    )

    return pos_new, mask_new
