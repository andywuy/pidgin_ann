import random
import os
import shutil
import zipfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def readzip(inp):
    '''
    Read the zipfile.

    Parameter
    ---------
    inp : str
        The path to the file.

    Return
    ------
    f1 : List[List[str]]
        A list of sublists. Each sublist contain: 
        [Uniprot, Organism, Compound_ID, Smiles, Measurement, Confidence_Score, Activity_Units, 
        Assay_ID, Orthologue_Flag, 100_Flag, 10_Flag, 1_Flag, 0.1_Flag]
    '''
    with zipfile.ZipFile(inp) as zfile:
        with zfile.open(inp.split('/')[-1].split('.zip')[0], 'r') as fid:
            f1 = fid.read().splitlines()[1:]
    random.seed(123)
    random.shuffle(f1)
    f1 = [i.decode('ascii').split('\t') for i in f1]
    f1.sort(key=lambda x: x[7], reverse=True)
    return f1


def molfp_file(query):
    '''
    Generate ECFP4 fingerprints for the given list of molecules.

    Parameters
    ----------
    query : List[List[str]]
        A list of sublists. 
        Each sublist contain: [Uniprot, Organism, Compound_ID, Smiles, Measurement, Confidence_Score, Activity_Units, 
        Assay_ID, Orthologue_Flag, 100_Flag, 10_Flag, 1_Flag, 0.1_Flag]  

    Return
    ------
    ret : List[matrix,scaffs,pids]
        matrix is a 2D matrix. Each row is the ECFP4 fingerprint. One fingerprint per molecule. 
        scaffs is a list of scaffolds. One scaffold per molecule.
        pids is a list of the Assay_IDs. One Assay_ID per molecule.
    '''
    matrix = []
    scaffs = []
    pids = []
    for row in query:
        smile = row[3]  # Get the smile of the molecule

        m = Chem.MolFromSmiles(smile)

        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        try:
            scaf = Chem.MolToSmiles(MakeScaffoldGeneric(m))
        except:
            scaf = ''
        matrix.append(list(map(int, list(fp.ToBitString()))))
        scaffs.append(scaf)
        pids.append(row[7])

    matrix = np.array(matrix, dtype=np.int8)
    ret = [matrix, scaffs, pids]
    return ret


class Model(object):
    '''
    Create the model inputs.
    One model input is generated for each protein target at a given IC50 cut-off.
    '''

    def __init__(self, dir_datasets: str, path_info: str,
                 dir_outputs: str, n: int):
        '''
        Parameters
        ----------
        dir_datasets : str
            Path to the directory of bioactivity datasets.
        path_info : str
            Path to the info file that contains all information of datasets available.
        dir_outputs : str
            Path to the folder that contains the curated npz file for each dataset, 
            which will be used during the training. 
        n : int
            The number of datasets selected. 

        '''
        self.datasets = dir_datasets
        self.info = path_info
        self.outputs = dir_outputs
        self.n = n

        if os.path.exists(dir_outputs):
            shutil.rmtree(dir_outputs)
        os.mkdir(dir_outputs)
        return

    def create(self):
        '''
        Retrieve the datasets information and write npz files.
        '''
        dic_info = self.process_info()

        # col_dict : key is the IC50 cutoff, value is the column number in the record.
        col_dict = {'100': 9, '10': 10, '1': 11, '0.1': 12}
        count = 0
        for filename, mids in dic_info.items():
            # Retrieve data for each dataset.
            # training_data is a list of sublists.
            # Each sublist contain:
            # [Uniprot, Organism, Compound_ID, Smiles, Measurement,
            # Confidence_Score, Activity_Units,
            # Assay_ID, Orthologue_Flag, 100_Flag, 10_Flag, 1_Flag, 0.1_Flag]
            training_data = readzip(os.path.join(self.datasets, filename))

            # get the fingerprints, scaffolds and publication ids
            ecfp4, scafs, pids = molfp_file(training_data)

            # Assign each unique scaffold with an ID
            scaf_dict = {sc: idx for idx, sc in enumerate(set(scafs))}
            # Create an array of scaffold id
            scafs = np.array([scaf_dict[s] for s in scafs])

            # create an array of publication id
            pids = np.array(pids)

            for mid in mids:
                # find the indexes of active and inactive compounds
                actives, inactives = [], []
                col = col_dict[mid.split('_')[-1]]
                for idx, line in enumerate(training_data):
                    if line[col] == '1':
                        actives.append(idx)
                    if line[col] == '0':
                        inactives.append(idx)

                # save the training input for each model_id
                np.savez_compressed(os.path.join(self.outputs, mid + '.npz'),  # file name
                                    # active compounds fingerprints
                                    actives=ecfp4[actives],
                                    # inactive compounds fingerprints
                                    inactives=ecfp4[inactives],
                                    # active compounds publication ids
                                    pids=pids[actives],
                                    # active compounds scaffolds
                                    scafs=scafs[actives])  
            count = count+1
            print('extracting data {:.4%} percent'.format(
                float(count)/len(dic_info)))

        print('Everything is done!')
        return

    def process_info(self):
        '''
        Process the information file.

        Return a dictionary where the key is the filename of a dataset, 
        and the value is a list of model_ids.
        The filename is in the form "[uniprot_id].smi.zip". 
        The model_id is in the form "[uniport_id],[SE]_[IC50]".
        [] indicates replacement field.
        uniprot_id is the ID of the protein in the UniProt database.
        SE indicates whether spherical exclusion is applied. It is an optional field.
        IC50 is the IC50 cutoff in micromole.        
        '''
        inf = []  # Store the tabulated information file.
        with open(self.info, 'r') as inp:
            for line in inp:
                inf.append(line.strip().split('\t'))

        inf = inf[1:]  # Skip the header line

        # Get a subset of models
        np.random.seed(self.n)
        inf_sub = []
        for i in np.random.choice(np.arange(len(inf)), size=self.n, replace=False):
            inf_sub.append(inf[i])

        # fs is a dictionary where the key is the model_id, 
        # and the value is the uniprot_id of the protein.
        fs = {line[-1]: line[0] for line in inf_sub}
        print('Total number of models = {}.'.format(len(fs)))

        # ret is a dict where the key is the filename of a dataset,
        # and the value is a list of model_ids.
        ret = dict()
        for mid, uniprot in fs.items():
            try:
                ret[uniprot + '.smi.zip'].append(mid)
            except:
                ret[uniprot + '.smi.zip'] = [mid]

        return ret
