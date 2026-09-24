import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from ._base import register_dataset

ALLOWED_AG_TYPES = {
 'peptide',
 'protein | peptide',
 'peptide | protein',
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein',
    'protein | protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0
                      
TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]
TEST_MODA_PATH = f'./test_moda_pdb_names.txt'
TEST_RABD_PATH = f'./test_radb_pdb_names.txt'
TEST_SAB23H2Ab_PATH = f'./test_sab23h2ab_pdb_names.txt'


def _dist_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_is_main_process():
    return (not _dist_is_initialized()) or torch.distributed.get_rank() == 0


def _dist_barrier():
    if _dist_is_initialized():
        torch.distributed.barrier()


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])                          


def get_cdr_indices(cdr_flag, cdr_type):
    
    cdr_mask = (cdr_flag == cdr_type)
    if not cdr_mask.any():                     
        return (None, None)
    indices = torch.where(cdr_mask)[0]
    return (indices[0].item(), indices[-1].item() + 1)                


                                                               
def PDB_scheme_to_parser(flag_size, pdb_seq_idx, chain_type, CDRDef):
    cdr_flag = torch.zeros(flag_size)
    fr_flag = torch.zeros(flag_size)

    for position, idx in pdb_seq_idx.items():
        resseq = position[1]
        cdr_type = CDRDef.to_cdr(chain_type, resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
            fr_flag[idx] = 0
        else:
            fr_type = CDRDef.to_fr(chain_type, resseq)
            assert fr_type is not None
            fr_flag[idx] = fr_type
            cdr_flag[idx] = 0
                                 
    cdr_flag_not = torch.where(cdr_flag >= 1, 0, 1).to(torch.bool)
    assert torch.equal(cdr_flag_not, (fr_flag > 0).to(torch.bool))
    return cdr_flag, fr_flag


                                          
def numbering_scheme_to_parser(data, chain_type, numbering_scheme, CDRDef):                           
    aa_seq = _aa_tensor_to_sequence(data)                   
    try:
        from anarci import anarci
        results = anarci([(chain_type, aa_seq)], scheme=numbering_scheme)
        numbering = results[0][0][0][0]                    
        use_anarci = numbering is not None

    except Exception as e:
        logging.warning(f"ANARCI failed on H chain: {e}")
        use_anarci = False

    aa_nmb_list = [aa for (_, aa) in numbering]                   
    aa_nmb_seq = ''.join(aa_nmb_list)

           
    aa_fv = aa_nmb_seq.replace('-', '')
    assert aa_fv in aa_seq,\
        f"Numbering sequence without gaps '{aa_fv}' not in original sequence '{aa_seq}'"
    fv_start = aa_seq.find(aa_fv)

    if fv_start == -1:
        raise ValueError(f"ANARCI Fv region '{aa_fv}' not found in sequence '{aa_seq}'")

    fv_end = fv_start + len(aa_fv)                           

           
    seq_len = len(aa_seq)
    cdr_flag = torch.zeros(seq_len, dtype=torch.long)
    fr_flag = torch.zeros(seq_len, dtype=torch.bool)

    seq_idx = fv_start         
    for (num, _), aa_aa in numbering:
                    
        if aa_aa == '-':
            continue

                                      
        while seq_idx < fv_end and aa_seq[seq_idx] != aa_aa:
            seq_idx += 1
        if seq_idx >= fv_end:
            raise ValueError(f"AA mismatch between numbering and sequence at {aa_aa}")

                     
        cdr_type = CDRDef.to_cdr(chain_type, num)
        if cdr_type is not None:
            cdr_flag[seq_idx] = cdr_type
        else:
            fr_flag[seq_idx] = True

        seq_idx += 1
    return cdr_flag, fr_flag


                                                       
def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30,
                           numbering_scheme='chothia'):                                                                  
    if data is None or seq_map is None:
        return data, seq_map

    if numbering_scheme.lower() == "chothia":
        CDRDef = constants.ChothiaCDRRange
    elif numbering_scheme.lower() == "imgt":
        CDRDef = constants.IMGTCDRRange
    else:
        raise ValueError(f"Unsupported numbering scheme: {numbering_scheme}")

                                                                 
                                                      
                                                                                                                                  
                              
                                                                                   
     
                               
     
                                                                        
     
                                                                        
                                                                                       
                                                                                       
                                                                                       
     
                                                               

           
                                 
    cdr_flag, fr_flag = PDB_scheme_to_parser(flag_size=data['aa'].shape, pdb_seq_idx=seq_map, chain_type='H',
                                             CDRDef=CDRDef)

    data['cdr_flag'] = cdr_flag                                       
    data['fr_flag'] = fr_flag
                                                                      
    data['H1_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H1])
    data['H2_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H2])
    data['H3_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.H3])

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()

             
                                                                         
                                
     
                                                            
                                                                            
                                                                            
                                                                            
     
                                                                               
                                                                               
                                                                               
     
                                                                                        
                                                                                        
                                                                                        

                                                                

                                                                                                                                                                      
                                                                                                        
                                                                                                                                                       
                                                                                                                                              
                                                                                                                                            
                                                                                                                                           

                          
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

                                
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30, numbering_scheme='chothia'):
    if data is None or seq_map is None:
        return data, seq_map

    if numbering_scheme.lower() == "chothia":
        CDRDef = constants.ChothiaCDRRange
    elif numbering_scheme.lower() == "imgt":
        CDRDef = constants.IMGTCDRRange
    else:
        raise ValueError(f"Unsupported numbering scheme: {numbering_scheme}")

                                                                 
                                                                                                                                   
     
                                                                        
                               
     
                                                            
                                                                            
                                                                            
                                                                            
     
                                                                        
                                                                                       
                                                                                       
                                                                                       
     
                                                               

           
                                 
    cdr_flag, fr_flag = PDB_scheme_to_parser(flag_size=data['aa'].shape, pdb_seq_idx=seq_map, chain_type='L',
                                             CDRDef=CDRDef)
    data['cdr_flag'] = cdr_flag                                       
    data['fr_flag'] = fr_flag

                                                                      
    data['L1_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L1])
    data['L2_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L2])
    data['L3_seq'] = _aa_tensor_to_sequence(data['aa'][cdr_flag == constants.CDR.L3])

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
     
             
                                                                         
                                
     
                                                            
                                                                               
                                                                               
                                                                               
     
                                                                                        
                                                                                        
                                                                                        
     
                                                                
     
                                                                                                        
                                                                                                                
                                                       
                                                                                                                                              
                                                                                                                                            
                                                                                                                                           

                          
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

                        
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


                                                                                    
def preprocess_sabdab_structure(task,
                                counters):                                                                                                             
    entry = task['entry']
    number_scheme = task['number_scheme']
    pdb_path = task['pdb_path']
                                             
    parser = PDBParser(QUIET=True)
                                
        
                         
                                                  
                   
                                                    
        
    parsed = {
        'id': entry['id'],                                       
        'heavy': None,                                                              
        'heavy_seqmap': None,                                       
        'light': None,                                                               
        'light_seqmap': None,                                       
        'antigen': None,                                                                
        'antigen_seqmap': None,
                                                                                                                
    }

    try:
                                           
                                                
                                              
              
                                   
                                                                   
                             
                                                                                          
                                                               
              

                    
                                              
                                                                                                                           
           

                                                              
                                                                                                                                             
        model = parser.get_structure(id, pdb_path)[
            0]                                                                                 
        if entry['H_chain'] is not None:
            (                                                             
                parsed['heavy'],
                                                                                                                                                            
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                                                                                                       
                model[entry['H_chain']],                                                                            
                max_resseq=constants.MAX_RESLEN[number_scheme]['H']                                      
            ))

        if entry['L_chain'] is not None:
            (                                                             
                parsed['light'],                  
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(                  
                model[entry['L_chain']],              
                max_resseq=constants.MAX_RESLEN[number_scheme]['L']                                      
            ))

        if parsed['heavy'] is None and parsed['light'] is None:                           
              
            counters['issue_heavy & light_miss'] += 1
              
            raise ValueError('Neither valid H-chain or L-chain is found.')

        if len(entry['ag_chains']) > 0:                                               
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'],        
                parsed['antigen_seqmap']        
            ) = parsers.parse_biopython_structure(
                chains)                                                                                  


    except (
            PDBExceptions.PDBConstructionException,
            parsers.ParsingException,
            KeyError,
            ValueError,
    ) as e:
          
        if isinstance(e, parsers.ParsingException):
            counters['issue_sequnk'] += 1
        elif isinstance(e, PDBExceptions.PDBConstructionException):                                     
            print('mmcif_parse: %s {%s}', pdb_path, str(e))
            counters['issue_file'] += 1
          
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None                               

    return parsed                                                                                                      


def save_test_df(df, test_indices, save_dir, data_name, data_from_path):
                    
    test_data_df = df.iloc[test_indices].reset_index(drop=True)

                                
    test_data_csv = os.path.join(save_dir, data_name + '.csv')                
    os.makedirs(os.path.dirname(test_data_csv), exist_ok=True)

              
    test_data_df.to_csv(test_data_csv, index=False, sep=',')

    if not data_from_path:               
        test_data_txt = os.path.join(save_dir, data_name + '.txt')           
        os.makedirs(os.path.dirname(test_data_txt), exist_ok=True)
                    
        with open(test_data_txt, 'w', encoding='utf-8') as f:
            for t_id in self.ref_test_ids:
                f.write(t_id + '\n')


class SAbDabDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)        

    def __init__(
            self,
            summary_path='./data/sabdab_summary_all.tsv',
            total_data_dir='./data/all_structures/chothia',
            processed_dir='./data/processed',
            split='train',
            data_number_scheme='chothia',
            split_seed=2022,
            transform=None,
                                      
            reset_strc=True,
            test_data='',                                         
            force_ref_test_ids=False,
            cluster_exclusion_use_ref_ids=False,
            test_cluster_source_processed_dir=None,
    ):
        super().__init__()
        self.split = split
        self.test_data = test_data
        self.force_ref_test_ids = force_ref_test_ids
        self.cluster_exclusion_use_ref_ids = cluster_exclusion_use_ref_ids
        self.test_cluster_source_processed_dir = test_cluster_source_processed_dir
        print(f'split {split};;test_data {test_data}')

        self.ref_test_ids = []
        self.fact_test_ids = []

        self.summary_path = summary_path
        self.total_data_dir = total_data_dir
        self.reset_strc = reset_strc
        assert total_data_dir.split('/')[
                   -1] == data_number_scheme, f'total_data_dir {total_data_dir}, number_scheme {data_number_scheme}'
        self.data_number_scheme = data_number_scheme
        if not os.path.exists(total_data_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {total_data_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.sabdab_entries = None                                                                                                     
        self._load_sabdab_entries()                                                                      

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset_strc)

        if self.split != 'test':
            self.clusters = None                                                                            
            self.id_to_cluster = None                                                                                  
            self._load_clusters(reset_strc)                                

        self.ids_in_split = None                                    
        self._load_split(split, split_seed)                 

        self.transform = transform

    def _load_sabdab_entries(self):                                                                   
        df = pd.read_csv(self.summary_path, sep='\t')               
        entries_all = []
        ref_test_entries = []
        ref_test_indices = []                  

        fact_test_entries = []
        fact_test_indices = []                    

          
        ckeck_test_info = []
          
                             
                       
                                 
                                      
                                                                        
                                    
                                             
                                            
                                          
                                    
         
                                          
                                                                          
                                    
                                             
                                            
                                          
                                      
                         
        if '.' in self.test_data:              
            with open(self.test_data, 'r', encoding='utf-8') as f:
                for line in f:
                                       
                    id = line.strip()
                    self.ref_test_ids.append(id)

        for i, row in tqdm(
                df.iterrows(),
                dynamic_ncols=True,
                desc='Loading entries',
                total=len(df),
        ):
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode=row['pdb'],
                H=nan_to_empty_string(row['Hchain']),
                L=nan_to_empty_string(row['Lchain']),
                Ag=''.join(split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                ))
            )
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {                                                    
                'id': entry_id,                                       
                'pdbcode': row['pdb'],        
                'number_scheme': self.data_number_scheme,
                'H_chain': nan_to_none(row['Hchain']),                   
                'L_chain': nan_to_none(row['Lchain']),                   
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

                       
                                                                                       
            if self.ref_test_ids:               
                if entry_id in self.ref_test_ids:
                    ref_test_entries.append(entry)
                    ref_test_indices.append(i)
            else:
                if entry['ag_name'] in TEST_ANTIGENS:
                    self.ref_test_ids.append(entry['id'])
                    ref_test_entries.append(entry)
                    ref_test_indices.append(i)

                                                                  
                                              
                                                    

                            

            if (                          
                    (entry['ag_type'] in ALLOWED_AG_TYPES or entry[
                        'ag_type'] is None)                                                       
                    and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
            ):
                                             
                                                                                                                                                
                
                entries_all.append(entry)
                                                
                if entry_id in self.ref_test_ids:
                    fact_test_entries.append(entry)
                    fact_test_indices.append(i)
                    self.fact_test_ids.append(entry['id'])

        print(f'===total entry{len(df)}, first clean csv {len(entries_all)}, filter {len(df) - len(entries_all)}')
        assert len(df) >= len(entries_all)

                     
        test_from_path = True if '.' in self.test_data else False
        save_test_df(df, ref_test_indices, self.processed_dir, 'ref_test_data', data_from_path=test_from_path)
        save_test_df(df, fact_test_indices, self.processed_dir, 'fact_test_data', data_from_path=test_from_path)

        if self.split == 'test':
            entries_all = ref_test_entries

        print(f'total entries_all {len(entries_all)}')
        self.sabdab_entries = entries_all                                                                     
          
        assert len(self.ref_test_ids) >= len(
            self.fact_test_ids), f"self.ref_test_ids {len(self.ref_test_ids)}, self.fact_test_ids {len(self.fact_test_ids)}"

        if len(self.ref_test_ids) > len(self.fact_test_ids):
            diff = [x for x in self.ref_test_ids if x not in self.fact_test_ids]
            print(diff)          
            for e_id in diff:
                for e in ref_test_entries:
                    if e_id == e['id']:
                        ckeck_test_info.append({e['id']: {"ag_type": e['ag_type'], "resolution": e['resolution']}})
                        break
            print(f"=============diff=============\n{diff}\n=============self.ref_test_ids=========\n{self.ref_test_ids}\n\
            ===========self.fact_test_ids=============\n{self.fact_test_ids}\n===================ckeck_test_info=========\n{ckeck_test_info}")

                                                                                                        
    def _load_structures(self,
                         reset_strc):                                                                              
        cache_ids_path = self._structure_cache_path + '-ids'
        cache_lock_path = self._structure_cache_path + '-lock'
        need_rebuild = (
            reset_strc
            or not os.path.exists(self._structure_cache_path)
            or not os.path.exists(cache_ids_path)
        )

        if _dist_is_main_process() and need_rebuild:
            for stale_path in (
                self._structure_cache_path,
                cache_ids_path,
                cache_lock_path,
            ):
                if os.path.exists(stale_path):
                    os.unlink(stale_path)
            self._preprocess_structures()

        _dist_barrier()

        if not os.path.exists(self._structure_cache_path) or not os.path.exists(cache_ids_path):
            raise FileNotFoundError(
                f"Structure cache missing in {self.processed_dir} after preprocessing."
            )

        with open(cache_ids_path, 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(                                                    
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    def _preprocess_structures(self):
        tasks = []
          
        cnt_total = len(self.sabdab_entries)
        cnt_notexist = 0
        notexist_pdb = []
          
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.total_data_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                  
                cnt_notexist += 1
                notexist_pdb.append(entry['pdbcode'])
                  
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({                                  
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
                'number_scheme': entry['number_scheme']
            })
          
        print(f'total {cnt_total};; not_exit pdb {cnt_notexist}; exit {len(tasks)}')
        with open('notexist_pdb.txt', 'w') as f:
            for item in notexist_pdb:
                f.write(str(item) + '\n')
          
                                                      
          
        from multiprocessing import Manager
        manager = Manager()
        counters = manager.dict({
            'issue_file': 0,
            'issue_sequnk': 0,
            'issue_heavy & light_miss': 0
        })

                                             
                                      
              
                               
                                                        
                         
                                                          
              
                    
                                                                       
                                                                                          
                                                                          
                                                                                           
                                                                          
                                                                                              
                                                                                                                                            
           
                                                              
                                                                                                                                             
                     
        data_list = joblib.Parallel(
                                                                                                                                                                                                                                     
            n_jobs=max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_sabdab_structure)(task, counters)                        
            for task in tqdm(tasks, dynamic_ncols=True,
                             desc='Preprocess (parse strcture [cdr_flag, fr_flag, H1-3 seqs, L1-3 seqs])')
        )
          
        num_filter = 0
          
                                                                                                
        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:                                              
                      
                    num_filter += 1
                      
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'),
                        pickle.dumps(data))                                                                             

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)

          
        print(f"Total {len(data_list)}, PDB_parsed error {num_filter} -> final {len(ids)}.\
         filter counter issues: issue_file {counters['issue_file']}, sequnk {counters['issue_sequnk']}, issue_heavy & light_miss' {counters['issue_heavy & light_miss']}")
                
          

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset_strc):                                                            
        need_rebuild = reset_strc or not os.path.exists(self._cluster_path)

        if _dist_is_main_process() and need_rebuild:
            cluster_prefix = os.path.join(self.processed_dir, 'cluster_result')
            cluster_tmp_dir = os.path.join(self.processed_dir, 'cluster_tmp')
            for suffix in ('', '.dbtype', '.index', '.lookup', '_cluster.tsv'):
                stale_path = cluster_prefix + suffix
                if os.path.exists(stale_path):
                    os.unlink(stale_path)
            if os.path.isdir(cluster_tmp_dir):
                subprocess.run(['rm', '-rf', cluster_tmp_dir], check=True)
            self._create_clusters()                                     

        _dist_barrier()

        if not os.path.exists(self._cluster_path):
            raise FileNotFoundError(
                f"Cluster file missing in {self.processed_dir} after mmseqs clustering."
            )

        clusters, id_to_cluster = {}, {}                                                                                      
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):                                                             
        cdr_records = []
           
        cnt_total = 0
        cnt_heavy = 0
        cnt_light = 0
        print(f'self.db_ids {len(self.db_ids)} {self.db_ids[:2]}...')
           
        seen_cluster_record_ids = set()
        for id in self.db_ids:
              
            cnt_total += 1
              
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                  
                cnt_heavy += 1
                  
                seen_cluster_record_ids.add(structure['id'])
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id=structure['id'],
                    name='',
                    description='',
                ))
            elif structure['light'] is not None:
                  
                cnt_light += 1
                  
                seen_cluster_record_ids.add(structure['id'])
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id=structure['id'],
                    name='',
                    description='',
                ))
        extra_cluster_dirs = self.test_cluster_source_processed_dir or []
        if isinstance(extra_cluster_dirs, str):
            extra_cluster_dirs = [p for p in extra_cluster_dirs.split(',') if p]
        for extra_dir in extra_cluster_dirs:
            extra_ids_path = os.path.join(extra_dir, 'structures.lmdb-ids')
            extra_lmdb_path = os.path.join(extra_dir, 'structures.lmdb')
            if not (os.path.exists(extra_ids_path) and os.path.exists(extra_lmdb_path)):
                continue
            with open(extra_ids_path, 'rb') as f:
                extra_ids = set(pickle.load(f))
            extra_db = lmdb.open(
                extra_lmdb_path,
                map_size=self.MAP_SIZE,
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with extra_db.begin() as txn:
                for id in self.ref_test_ids:
                    if id in seen_cluster_record_ids or id not in extra_ids:
                        continue
                    payload = txn.get(id.encode())
                    if payload is None:
                        continue
                    structure = pickle.loads(payload)
                    if structure.get('heavy') is not None:
                        seq = structure['heavy']['H3_seq']
                    elif structure.get('light') is not None:
                        seq = structure['light']['L3_seq']
                    else:
                        continue
                    seen_cluster_record_ids.add(structure['id'])
                    cdr_records.append(SeqRecord.SeqRecord(
                        Seq.Seq(seq),
                        id=structure['id'],
                        name='',
                        description='',
                    ))
            extra_db.close()

        missing_cluster_ids = [
            id for id in self.ref_test_ids
            if id not in seen_cluster_record_ids
        ]
        if missing_cluster_ids:
            print(f'test ids missing from cluster records: {missing_cluster_ids}')

        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path,
                    'fasta')                                                                       
          
        print(f'Totally data {cnt_total}; heavy cdr {cnt_heavy}; light cdr3 {cnt_light} ')
          

        cmd = ' '.join([
                                                           
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),        
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',         
            '-c', '0.8',                     
            '--cov-mode', '1',        
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)
                

    def _load_split(self, split, split_seed):                                                                 
        assert split in ('train', 'val', 'test')
                            
                   
                                                                
                            
                                     
                                    
                                  
         
                         
                         
                                              
                                         
           
                           
                           
                                                
                                                
             
         
                          
                   
                                                                  
                            
                                     
                                    
                                  
         
                         
                         
                                              
                                         
           
                                    
                           
                                                
                                                  
             

        if split == 'test':
            if self.force_ref_test_ids:
                missing = [id for id in self.ref_test_ids if id not in self.db_ids]
                if missing:
                    raise RuntimeError(
                        f'Requested test IDs missing from parsed DB: {missing}'
                    )
                self.ids_in_split = list(self.ref_test_ids)
                print(
                    f'force_ref_test_ids active: test complexes {len(self.ids_in_split)}'
                )
            else:
                self.ids_in_split = self.fact_test_ids
        else:
            cluster_source_test_ids = (
                self.ref_test_ids
                if self.cluster_exclusion_use_ref_ids
                else self.fact_test_ids
            )
            cluster_source_test_ids = [
                id for id in cluster_source_test_ids
                if id in self.id_to_cluster
            ]
            test_relevant_clusters = set(
                [self.id_to_cluster[id] for id in cluster_source_test_ids])                                        
            print(
                f'factual test complex {len(self.fact_test_ids)};;cluster source {len(cluster_source_test_ids)};;test_clusters {len(test_relevant_clusters)}')

                                           
            ids_train_val = [                                          
                entry['id']
                for entry in self.sabdab_entries
                if self.id_to_cluster[entry['id']] not in test_relevant_clusters
            ]
            random.Random(split_seed).shuffle(ids_train_val)               

            self.ids_in_split = ids_train_val

            if split == 'val':                 
                self.ids_in_split = ids_train_val[:20]
            else:
                self.ids_in_split = ids_train_val[20:]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        return data                   


@register_dataset('sabdab')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDataset(
        summary_path=cfg.summary_path,
        total_data_dir=cfg.total_data_dir,
        processed_dir=cfg.processed_dir,
        split=cfg.split,
        split_seed=cfg.get('split_seed', 2022),
        transform=transform,
        test_data=cfg.test_data,
        reset_strc=cfg.reset_strc,
        data_number_scheme=cfg.data_number_scheme,
        force_ref_test_ids=cfg.get('force_ref_test_ids', False),
        cluster_exclusion_use_ref_ids=cfg.get('cluster_exclusion_use_ref_ids', False),
        test_cluster_source_processed_dir=cfg.get('test_cluster_source_processed_dir', None)
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset_strc', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset_strc:
        sure = input('Sure to reset_strc? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split,
        reset_strc=args.reset_strc
    )
    print(dataset[0])
    print(f"len(dataset) {len(dataset)}, len(dataset.clusters) {len(dataset.clusters)}")
