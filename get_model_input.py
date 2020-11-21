### Extract the data from the bioactivity dataset and generate the inputs for the training.
### The code is adapted from the original PIDGINv4 code and it is written in python 2.

#libraries
import random
import zipfile
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from rdkit import RDLogger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def readzip(inp):
    with zipfile.ZipFile(inp) as zfile:
        with zfile.open(inp.split('/')[-1].split('.zip')[0], 'r') as fid:
            f1 = fid.read().splitlines()[1:]
    random.seed(123)
    random.shuffle(f1)
    f1 = [i.split('\t') for i in f1]
    f1.sort(key=lambda x: x[7], reverse=True)
    return f1

def molfp(smile):
    """Get the morgan fingerprints and scaffold"""
    m = Chem.MolFromSmiles(smile)
    if m is None:
        return 500  # if smile is broken, return True so that we can use this in later functions
    else:
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m,2, nBits=2048)
        fp3 = AllChem.GetMorganFingerprintAsBitVect(m,3, nBits=2048)
        fp4 = AllChem.GetMorganFingerprintAsBitVect(m,2, useFeatures=True, nBits=2048)
        fp5 = AllChem.GetMorganFingerprintAsBitVect(m,3, useFeatures=True, nBits=2048)
        try: scaf = Chem.MolToSmiles(MakeScaffoldGeneric(m))
        except: scaf = ''
    return map(int,list(fp2.ToBitString())),map(int,list(fp3.ToBitString())), map(int,list(fp4.ToBitString())), map(int,list(fp5.ToBitString())), scaf

#generate rdkit ECFP and GOBBI fp per smile
def molfp_file(query):
    matrix4 = np.zeros((len(query), 2048), dtype=np.uint8)
    matrix6 = np.zeros((len(query), 2048), dtype=np.uint8)
    matrix4feat = np.zeros((len(query), 2048), dtype=np.uint8)
    matrix6feat = np.zeros((len(query), 2048), dtype=np.uint8)
    scaffs = []
    pids = []
    current_end = 0
    for row in query:
            if molfp(row[3]) == 500:
                print(row)
                continue
            else:
                result = molfp(row[3])
                matrix4[current_end] = np.array(result[0],dtype=np.uint8)
                matrix6[current_end] = np.array(result[1],dtype=np.uint8)
                matrix4feat[current_end] = np.array(result[2],dtype=np.uint8)
                matrix6feat[current_end] = np.array(result[3],dtype=np.uint8)
                scaffs.append(result[4])
                pids.append(row[7])
                current_end += 1
    return matrix4,matrix6,matrix4feat,matrix6feat,scaffs,pids

#process inf file
def calculateinfs():
    ret = dict()
    temp = open(ipath).read().splitlines()
    inf = [line.split('\t') for line in temp]
    # skip header
    inf = inf[1:]
    #get a 1000 subset of all models.
    np.random.seed(123)
    inf_sub = []
    for i in np.random.choice(np.arange(len(inf)),size=1000,replace=False):
        inf_sub.append(inf[i]) 
    fs = {line[-1]:line[0] for line in inf_sub}
    print('No. of total models: ' + str(len(fs)))
    mids_lst = fs.keys()
    with open(off+"mids_list.txt", "w") as f:
        for s in mids_lst:
            f.write(s +"\n")
        f.close()
    
    for mid,uniprot in fs.iteritems():
        try: ret[uniprot + '.smi.zip'].append(mid)
        except: ret[uniprot + '.smi.zip'] = [mid]
    print('No. of uniprot.smi.zip files to do (not unique models): ' + str(len(ret)))
    return ret

def do_f(mod_chunk):
    col_dict = {'100':9,'10':10,'1':11,'0.1':12}
    count = 0
    for chunk in mod_chunk:
        count = count+1
        filename,mids = chunk
        training_data = readzip(dpath + filename)
        # get the fingerprints, scaffolds and publication ids 
        ecfp4,ecfp6,fcfp4,fcfp6,scafs,pids = molfp_file(training_data)
        # create the dictionary to look at the scaffold id
        scaf_dict = {sc:idx for idx,sc in enumerate(set(scafs))}
        scafs = np.array([scaf_dict[s] for s in scafs])
        pids = np.array(pids)
        for mid in mids:
            col = col_dict[mid.split('_')[-1]]
            actives, inactives = [], []
            for idx, line in enumerate(training_data):
                if line[col] == '1': actives.append(idx)
                if line[col] == '0': inactives.append(idx)
            fps = ecfp4
            # save the input in npz
            # use fp_name, active compounds, publication id, scaffolds, inactive compounds, model id to train
            np.savez_compressed(off + mid + '.npz',
                                actives = fps[actives],
                                inactives = fps[inactives],
                                pids = pids[actives],
                                scafs = scafs[actives])
        print('extracting data %.4f percent' % (float(count)/len(mod_chunk)*100))
    return

    
#process inf files:
def do_fs(inp):
    tasks = [[filename,cols] for filename,cols in inp.iteritems()]
    do_f(tasks)
    return 
    
if __name__ == "__main__":
    dpath = raw_input('Input directory for the bioactivity dataset (e.g. /no_ortho/bioactivity_dataset/):')
    ipath = raw_input('Input path to the information file (e.g. /no_ortho/classes_in_model_no_ortho.txt):')
    # The output directory for the model inputs
    off = './model_inputs/'
    # Get the list of 1000 models.
    fs = calculateinfs() 
    # Prepare the inputs (stored in .npz) for the training
    do_fs(fs)
    print('Everything is done!')
    
