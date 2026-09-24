import torch
import random
from typing import List, Optional

from ..protein import constants
from ._base import register_transform
from copy import deepcopy as dcp


def random_shrink_extend(flag, min_length=5, shrink_limit=1, extend_limit=2):
    first, last = continuous_flag_to_range(flag)                   
    length = flag.sum().item()             
    if (length - 2 * shrink_limit) < min_length:
        shrink_limit = 0
                                                    
    first_ext = max(0, first - random.randint(-shrink_limit, extend_limit))
    last_ext = min(last + random.randint(-shrink_limit, extend_limit), flag.size(0) - 1)
    flag_ext = flag.clone()
    flag_ext[first_ext: last_ext + 1] = True
    return flag_ext


def continuous_flag_to_range(flag):
    first = (torch.arange(0, flag.size(0))[flag]).min().item()                               
    last = (torch.arange(0, flag.size(0))[flag]).max().item()
    return first, last                    


@register_transform('mask_single_cdr')
class MaskSingleCDR(object):                       

    def __init__(self, selection=None, augmentation=None):                       
        super().__init__()
        assert augmentation is not None
                                                             
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
            'CDR3': 'CDR3',                                 
        }
        assert selection is None or selection in cdr_str_to_enum
        self.selection = cdr_str_to_enum.get(selection, None)
        self.augmentation = augmentation

    def perform_masking_(self, data, selection=None):
        cdr_flag = data['cdr_flag']

        if selection is None:                                 
            cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()
            cdr_to_mask = random.choice(cdr_all)
        else:                    
            cdr_to_mask = selection

        cdr_to_mask_flag = (cdr_flag == cdr_to_mask)
        if self.augmentation:
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag)

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
        left_idx = max(0, cdr_first - 1)
        right_idx = min(data['aa'].size(0) - 1, cdr_last + 1)
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True
        anchor_flag[right_idx] = True

        data['generate_flag'] = cdr_to_mask_flag
        data['anchor_flag'] = anchor_flag
          
        data['fix_cdr_flag'] = dcp(data['cdr_flag'])
        data['fix_cdr_flag'][cdr_flag == cdr_to_mask] = 0

    def __call__(self, structure):                       
        if self.selection is None:                          
            ab_data = []
            if structure['heavy'] is not None:
                ab_data.append(structure['heavy'])
            if structure['light'] is not None:
                ab_data.append(structure['light'])
            data_to_mask = random.choice(ab_data)               
            sel = None                  
        elif self.selection in (constants.CDR.H1, constants.CDR.H2, constants.CDR.H3,):             
            data_to_mask = structure['heavy']          
            sel = int(self.selection)          
        elif self.selection in (constants.CDR.L1, constants.CDR.L2, constants.CDR.L3,):             
            data_to_mask = structure['light']          
            sel = int(self.selection)          
        elif self.selection == 'CDR3':                                                       
            if structure['heavy'] is not None:
                data_to_mask = structure['heavy']
                sel = constants.CDR.H3
            else:
                data_to_mask = structure['light']
                sel = constants.CDR.L3

        self.perform_masking_(data_to_mask, selection=sel)

                                                            
        if structure['heavy'] is not None:
            if 'fix_cdr_flag' not in structure['heavy']:
                structure['heavy']['fix_cdr_flag'] = dcp(structure['heavy']['cdr_flag'])
        if structure['light'] is not None:
            if 'fix_cdr_flag' not in structure['light']:
                structure['light']['fix_cdr_flag'] = dcp(structure['light']['cdr_flag'])

        return structure


@register_transform('mask_multiple_cdrs')
class MaskMultipleCDRs(object):                          

    def __init__(self, selection: Optional[List[str]] = None, augmentation=None):                         
        super().__init__()
        assert augmentation is not None
                                                                
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
        }
        if selection is not None:            
            self.selection = [cdr_str_to_enum[s] for s in selection]
        else:
            self.selection = None
        self.augmentation = augmentation

    def mask_one_cdr_(self, data, cdr_to_mask):
        cdr_flag = data['cdr_flag']

        cdr_to_mask_flag = (cdr_flag == cdr_to_mask)                           
        if self.augmentation:               
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag)

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
        left_idx = max(0, cdr_first - 1)                                              
        right_idx = min(data['aa'].size(0) - 1, cdr_last + 1)                                                
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True            
        anchor_flag[right_idx] = True            

        if 'generate_flag' not in data:
            data['generate_flag'] = cdr_to_mask_flag
            data['anchor_flag'] = anchor_flag

        else:                                                            
            data['generate_flag'] |= cdr_to_mask_flag
            data['anchor_flag'] |= anchor_flag

        if "fix_cdr_flag" not in data:
            data['fix_cdr_flag'] = dcp(data['cdr_flag'])

        data['fix_cdr_flag'][cdr_flag == cdr_to_mask] = 0

    def mask_for_one_chain_(self, data):                   
        cdr_flag = data[
            'cdr_flag']                                                                                                                                           
        cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()                                     

        num_cdrs_to_mask = random.randint(1, len(cdr_all))                              

        if self.selection is not None:                               
            cdrs_to_mask = list(set(cdr_all).intersection(self.selection))
        else:                                                          
            random.shuffle(cdr_all)                 
            cdrs_to_mask = cdr_all[:num_cdrs_to_mask]

        for cdr_to_mask in cdrs_to_mask:                                                                             
            self.mask_one_cdr_(data, cdr_to_mask)                                             

    def __call__(self, structure):
        if structure['heavy'] is not None:
            self.mask_for_one_chain_(structure['heavy'])

        if structure['light'] is not None:
            self.mask_for_one_chain_(structure['light'])

        return structure


@register_transform('mask_semantic_coalition')
class MaskSemanticCoalition(object):
    

    _TYPE_IDS = {'all': 0, 'leave_one': 1, 'h3_only': 2, 'subset': 3, 'h3_l1': 4}

    def __init__(self, mode='sample', selection=None):
        super().__init__()
        if mode not in ('sample', 'random_multi_cdrs', 'all', 'leave_one', 'h3_only', 'subset', 'h3_l1'):
            raise ValueError('Unknown semantic coalition mode: %s' % mode)
        self.mode = mode
        cdr_names = {
            'H1': constants.CDR.H1, 'H2': constants.CDR.H2, 'H3': constants.CDR.H3,
            'L1': constants.CDR.L1, 'L2': constants.CDR.L2, 'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1, 'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3, 'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2, 'L_CDR3': constants.CDR.L3,
        }
        self.selection = None if selection is None else [
            int(cdr_names.get(name, name)) for name in selection
        ]

    @staticmethod
    def _available(structure):
        available = []
        for name in ('heavy', 'light'):
            data = structure.get(name)
            if data is None:
                continue
            available.extend(
                int(x) for x in data['cdr_flag'].unique().tolist() if int(x) > 0
            )
        return sorted(set(available))

    def _select(self, structure):
        available = self._available(structure)
        if not available:
            return [], 'all'
        if self.selection is not None:
            full_available = list(available)
            selected = sorted(set(available).intersection(self.selection))
            if not selected:
                return [], 'all'
                                                                             
                                                                               
                                                                             
            if selected == full_available:
                selected_mode = 'all'
            elif selected == sorted(set((int(constants.CDR.H3), int(constants.CDR.L1))).intersection(full_available)) and len(selected) == 2:
                selected_mode = 'h3_l1'
            else:
                selected_mode = 'subset'
            return selected, selected_mode

        mode = self.mode
        if mode in ('sample', 'random_multi_cdrs'):
                                                                               
                                                                               
                                                                              
                                                                          
                                                                                 
            selected = []
            heavy_ids = [cdr_id for cdr_id in available if cdr_id in (
                int(constants.CDR.H1), int(constants.CDR.H2), int(constants.CDR.H3)
            )]
            light_ids = [cdr_id for cdr_id in available if cdr_id in (
                int(constants.CDR.L1), int(constants.CDR.L2), int(constants.CDR.L3)
            )]
            for chain_ids in (heavy_ids, light_ids):
                if not chain_ids:
                    continue
                count = random.randint(1, len(chain_ids))
                selected.extend(random.sample(chain_ids, count))
            selected = sorted(selected)
            return selected, ('all' if selected == available else 'subset')

        if mode == 'all':
            selected = available
        elif mode == 'subset':
                                                                                
                                                                          
                                                               
            if len(available) <= 1:
                selected = available
                mode = 'all'
            else:
                subset_size = random.randint(1, len(available) - 1)
                selected = sorted(random.sample(available, subset_size))
        elif mode == 'leave_one':
            removed = random.choice(available)
            selected = [cdr_id for cdr_id in available if cdr_id != removed]
            if not selected:
                selected = available
                mode = 'all'
        elif mode == 'h3_l1':
            requested = (int(constants.CDR.H3), int(constants.CDR.L1))
            selected = sorted(set(available).intersection(requested))
            if len(selected) < 2:
                                                                            
                                                                         
                h3 = int(constants.CDR.H3)
                selected = [h3] if h3 in available else selected
                mode = 'h3_only' if selected else 'all'
                if not selected:
                    selected = available
        else:
            h3 = int(constants.CDR.H3)
            l3 = int(constants.CDR.L3)
            selected = [h3] if h3 in available else ([l3] if l3 in available else available)
            if selected == available and h3 not in available and l3 not in available:
                mode = 'all'
        return selected, mode

    @staticmethod
    def _apply_to_chain(data, selected):
        cdr_flag = data['cdr_flag']
        generate_flag = torch.zeros_like(cdr_flag, dtype=torch.bool)
        anchor_flag = torch.zeros_like(generate_flag)
        for cdr_id in selected:
            cdr_mask = cdr_flag == cdr_id
            if not cdr_mask.any():
                continue
            generate_flag |= cdr_mask
            indices = torch.where(cdr_mask)[0]
            anchor_flag[max(0, int(indices[0]) - 1)] = True
            anchor_flag[min(data['aa'].size(0) - 1, int(indices[-1]) + 1)] = True

        data['generate_flag'] = generate_flag
        data['anchor_flag'] = anchor_flag
        data['fix_cdr_flag'] = cdr_flag.clone()
        for cdr_id in selected:
            data['fix_cdr_flag'][cdr_flag == cdr_id] = 0

    def __call__(self, structure):
        selected, mode = self._select(structure)
        if structure.get('heavy') is not None:
            self._apply_to_chain(structure['heavy'], selected)
        if structure.get('light') is not None:
            self._apply_to_chain(structure['light'], selected)
        structure['coalition_type'] = self._TYPE_IDS[mode]
        return structure


@register_transform('mask_antibody')
class MaskAntibody(object):

    def mask_ab_chain_(self, data):
        data['generate_flag'] = torch.ones(data['aa'].shape, dtype=torch.bool)

    def __call__(self, structure):
        pos_ab_alpha = []
        if structure['heavy'] is not None:
            self.mask_ab_chain_(structure['heavy'])
            pos_ab_alpha.append(
                structure['heavy']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        if structure['light'] is not None:
            self.mask_ab_chain_(structure['light'])
            pos_ab_alpha.append(
                structure['light']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        pos_ab_alpha = torch.cat(pos_ab_alpha, dim=0)             

        if structure['antigen'] is not None:
            pos_ag_alpha = structure['antigen']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            ag_ab_dist = torch.cdist(pos_ag_alpha, pos_ab_alpha)                
            nn_ab_dist = ag_ab_dist.min(dim=1)[0]          
            contact_flag = (nn_ab_dist <= 6.0)          
            if contact_flag.sum().item() == 0:
                contact_flag[nn_ab_dist.argmin()] = True

            anchor_idx = torch.multinomial(contact_flag.float(), num_samples=1).item()
            anchor_flag = torch.zeros(structure['antigen']['aa'].shape, dtype=torch.bool)
            anchor_flag[anchor_idx] = True
            structure['antigen']['anchor_flag'] = anchor_flag
            structure['antigen']['contact_flag'] = contact_flag

        return structure


@register_transform('remove_antigen')
class RemoveAntigen:

    def __call__(self, structure):
        structure['antigen'] = None
        structure['antigen_seqmap'] = None
        return structure
