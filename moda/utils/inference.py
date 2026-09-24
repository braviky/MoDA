import torch
from .protein import constants


def find_cdrs(structure):
    cdrs = []
    if structure['heavy'] is not None:
        flag = structure['heavy']['cdr_flag']
        if int(constants.CDR.H1) in flag:
            cdrs.append('H_CDR1')
        if int(constants.CDR.H2) in flag:
            cdrs.append('H_CDR2')
        if int(constants.CDR.H3) in flag:
            cdrs.append('H_CDR3')

    if structure['light'] is not None:
        flag = structure['light']['cdr_flag']
        if int(constants.CDR.L1) in flag:
            cdrs.append('L_CDR1')
        if int(constants.CDR.L2) in flag:
            cdrs.append('L_CDR2')
        if int(constants.CDR.L3) in flag:
            cdrs.append('L_CDR3')
    
    return cdrs               


def get_residue_first_last(data):
    loop_flag = data['generate_flag']
    loop_idx = torch.arange(loop_flag.size(0))[loop_flag]
    idx_first, idx_last = loop_idx.min().item(), loop_idx.max().item()
    residue_first = (data['chain_id'][idx_first], data['resseq'][idx_first].item(), data['icode'][idx_first])
    residue_last = (data['chain_id'][idx_last], data['resseq'][idx_last].item(), data['icode'][idx_last])
    return residue_first, residue_last

      
def get_residue_first_last_multi(data):
    generate_flag = data['generate_flag']      
    cdr_flag = data['cdr_flag']                                                 

    idx_all = torch.arange(len(generate_flag))
    idx_gen = idx_all[generate_flag]

    if idx_gen.numel() == 0:
        return [None] * 6          

                              
    split_indices = torch.where(torch.diff(idx_gen) > 1)[0] + 1
    splits = torch.tensor_split(idx_gen, split_indices)
    print(f'splits {splits}')

                                        
    cdr_ranges = [None] * 6

    for seg in splits:
        start_idx = seg[0].item()
        end_idx = seg[-1].item()

                    
        seg_cdr_flags = cdr_flag[seg]
        seg_cdr_nonzero = seg_cdr_flags[seg_cdr_flags != 0]
        print(f'seg_cdr_flags {seg_cdr_flags};;seg_cdr_nonzero {seg_cdr_nonzero}')
        if len(seg_cdr_nonzero) == 0:
            continue                

                       
        cdr_id = int((torch.mode(seg_cdr_nonzero).values.item()))              
        idx_in_list = cdr_id - 1          
        print(f'cdr_id {cdr_id}, idx_in_list {idx_in_list}')
                          
        if cdr_ranges[idx_in_list] is not None:
            continue

        residue_first = (
            data['chain_id'][start_idx],
            data['resseq'][start_idx].item(),
            data['icode'][start_idx]
        )
        residue_last = (
            data['chain_id'][end_idx],
            data['resseq'][end_idx].item(),
            data['icode'][end_idx]
        )

        cdr_ranges[idx_in_list] = [residue_first, residue_last]

    return cdr_ranges




class RemoveNative(object):                                                                                             

    def __init__(self, remove_structure, remove_sequence):
        super().__init__()
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence

    def __call__(self, data):
        generate_flag = data['generate_flag'].clone()
                                                                                    
                                                                               
        if 'native_aa' not in data:
            data['native_aa'] = data['aa'].clone()
        if self.remove_sequence:                                           
            data['aa'] = torch.where(
                generate_flag, 
                torch.full_like(data['aa'], fill_value=int(constants.AA.UNK)),             
                data['aa']
            )

        if self.remove_structure:
            data['pos_heavyatom'] = torch.where(
                generate_flag[:, None, None].expand(data['pos_heavyatom'].shape),
                torch.randn_like(data['pos_heavyatom']) * 10,                             
                data['pos_heavyatom']
            )

        return data
