from shutil import rmtree
import time
import os
import math
import warnings
from dataclasses import dataclass, asdict

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf

from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc
from nonconformist.base import ClassifierAdapter

from rdkit.ML.Scoring.Scoring import CalcBEDROC

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import confusion_matrix, brier_score_loss, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import average_precision_score  # for the prauc score

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')  # suppress warning from tf

@dataclass
class Log:
    '''
    The dataclass containing the model information and performance metrics.
    '''
    model_id: str = ""
    fingerprint_type: str = ""
    num_scaffolds: int = None
    num_publications: int = None
    train_time_sec: float = None

    test_weighted_accuracy: float = None
    test_mcc: float = None
    test_roc: float = None
    test_prauc: float = None
    test_brier: float = None
    test_bedroc: float = None
    # The averaged predicted probability of being active for inactive compounds
    test_avg_prob_inactive: float = None
    # The standard deviation of predicted probability of being active for inactive compounds
    test_std_prob_inactive: float = None
    # The averaged predicted probability of being active for active compounds
    test_avg_prob_active: float = None
    # The standard deviation of predicted probability of being active for active compounds
    test_std_prob_active: float = None
    test_precision_inactive: float = None
    test_recall_inactive: float = None
    test_f1_inactive: float = None
    test_precision_active: float = None
    test_recall_active: float = None
    test_f1_active: float = None

    train_weighted_accuracy: float = None
    train_mcc: float = None
    train_roc: float = None
    train_prauc: float = None
    train_brier: float = None
    train_bedroc: float = None
    train_avg_prob_inactive: float = None
    train_std_prob_inactive: float = None
    train_avg_prob_active: float = None
    train_std_prob_active: float = None
    train_precision_inactive: float = None
    train_recall_inactive: float = None
    train_f1_inactive: float = None
    train_precision_active: float = None
    train_recall_active: float = None
    train_f1_active: float = None

    # LEAVE 50% OF PUBS OUT and do group splitting
    pub_avg_bedroc: float = None
    pub_std_bedroc: float = None
    pub_avg_roc: float = None
    pub_std_roc: float = None
    pub_avg_prauc: float = None
    pub_std_prauc: float = None
    pub_size: str = ""  # Sizes of each group

    # LEAVE 50% OF scaffolds OUT and do group splitting
    scaf_avg_bedroc: float = None
    scaf_std_bedroc: float = None
    scaf_avg_roc: float = None
    scaf_std_roc: float = None
    scaf_avg_prauc: float = None
    scaf_std_prauc: float = None
    scaf_size: str = ""

    # Time series split
    tscv_avg_bedroc: float = None
    tscv_std_bedroc: float = None
    tscv_avg_roc: float = None
    tscv_std_roc: float = None
    tscv_avg_prauc: float = None
    tscv_std_prauc: float = None
    tscv_size: str = ""

    # Conformal prediction
    # The proportion of compounds that are assigned to one class only (either active or inactive).
    one_class_rate: float = None
    # The proportion of compounds that not only are predicted to be active by the conformal prediction, but also have correct labels.
    true_active_rate: float = None
    # The proportion of compounds that not only are predicted to be inactive by the conformal prediction, but also have correct labels.
    true_inactive_rate: float = None


def train_dev_test_split(actives, pids, scafs, inactives, test_size, dev_size, min_size, random_seed):
    '''
    Split the data into training_set, development_set and test_set

    Parameters
    ----------
    actives : 2D array
        Each row is the 2048 bit molecular fingerprint of an active compound. 
    inactives : 2D array
        Each row is the 2048 bit molecular fingerprint of an inactive compound. 
    pids : 1D array, 
        It contains the publication_id for each active compound.
    scafs : 1D array
        It contains the scaffold_id for each active compound.
    test_size : float
        The proportion of the test_set.
    dev_size : float 
        The proportion of the development_set.
    min_size : int
        The minimum number of compounds in each set.
    random_seed : int
        The random seed.

    Return
    ------
    enough_compounds : bool
        Check if there are enough compounds in all three sets.
        If either actives or inactives does not have enough compounds, then enough_compounds is set to False.
    dic_compounds : Dict
        A dictionary of the splited data.
    '''
    dic_compounds = {}
    enough_compounds = True
    np.random.seed(random_seed)

    for i in ["active", "inactive"]:
        if i == "active":
            current = actives
        else:
            current = inactives

        num_compounds = len(current)
        idx = np.random.permutation(num_compounds)
        num_test = math.floor(num_compounds*test_size)
        num_dev = math.floor(num_compounds*dev_size)
        test_idx = idx[0:num_test]
        dev_idx = idx[num_test:num_dev+num_test]
        train_idx = idx[num_dev+num_test:]

        # if enough_compounds is False once, then it remainds False
        enough_compounds = enough_compounds and (len(test_idx) >= min_size and len(
            dev_idx) >= min_size and len(train_idx) >= min_size)

        if enough_compounds:
            if i == "active":
                dic_compounds['pids_train'] = pids[train_idx]
                dic_compounds['scafs_train'] = scafs[train_idx]
                dic_compounds['actives_train'] = current[train_idx]
                dic_compounds['actives_train_y'] = np.ones(
                    len(train_idx)).reshape(-1, 1)  # reshape into 2d vector
                dic_compounds['pids_dev'] = pids[dev_idx]
                dic_compounds['scafs_dev'] = scafs[dev_idx]
                dic_compounds['actives_dev'] = current[dev_idx]
                dic_compounds['actives_dev_y'] = np.ones(
                    len(dev_idx)).reshape(-1, 1)
                dic_compounds['actives_test'] = current[test_idx]
                dic_compounds['actives_test_y'] = np.ones(
                    len(test_idx)).reshape(-1, 1)
            else:
                dic_compounds['inactives_train'] = current[train_idx]
                dic_compounds['inactives_train_y'] = np.zeros(
                    len(train_idx)).reshape(-1, 1)
                dic_compounds['inactives_dev'] = current[dev_idx]
                dic_compounds['inactives_dev_y'] = np.zeros(
                    len(dev_idx)).reshape(-1, 1)
                dic_compounds['inactives_test'] = current[test_idx]
                dic_compounds['inactives_test_y'] = np.zeros(
                    len(test_idx)).reshape(-1, 1)

    return dic_compounds, enough_compounds

def write_training_log(log: Log, path: str):
    '''
    Writing trainlog.

    Parameters
    ----------
    log : Log
        It contains MODEL_ID and performance metrics. For details, see Class Log.

    path : str
        Path to the training log directory.
    '''
    path = os.path.join(path , 'training_log.txt')
    cols = list(log.__annotations__.keys())  # columns in the training log.
    dic = asdict(log)  # A dictionary {column_name : value}
    output = []  # The line that will be written to the training log.
    sep = '\t'  # The separator in the training log.
    for col in cols:
        val = dic[col]
        if val == None:
            output.append(" ")
        elif type(val) == str:
            output.append(val)
        elif type(val) == int:
            output.append(str(val))
        else:
            # otherwise it is float point number
            # keep 3 dp
            output.append('{:.3f}'.format(val))

    if not os.path.exists(path):
        # if the file does not exist, create the file, write the header, and append the first log.
        # prepare the header string
        header = sep.join(cols) + '\n'
        with open(path, "w") as logf:
            logf.write(header)
            logf.write(sep.join(output) + '\n')
    else:
        # if the file already exists, append the log to the end.
        with open(path, "a") as logf:
            logf.write(sep.join(output) + '\n')
    return

def get_model(input_shape, model_weights=None):
    '''
    Return the ANN model.

    Parameters
    ----------
    input_shape : Tuple
        The shape of the feature vector.
    model_weights : 
        The model weights.

    Return
    ------
    model : Sequential model.
    '''
    model = Sequential([Dense(100, input_shape=input_shape, activation='relu'),
                        BatchNormalization(),
                        Dense(10, activation='relu'),
                        Dense(1, activation='sigmoid')])
    if model_weights != None:
        model.set_weights(model_weights)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model

def fit_model(model, x, y):
    '''
    Fit the Sequential model.

    Parameters
    ----------
    model : Sequential model.
    x : 2D array
        Matrix of feature vectors.
    y : 1D array
        True labels.
    '''
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=3)
    sw = compute_sample_weight('balanced', y)
    model.fit(x, y, sample_weight=sw,
              batch_size=128, epochs=100,
              verbose=0, callbacks=[early_stopping])
    return

def calculate_performance(actives_ypredict, inactives_ypredict):
    '''
    Calculate performance metrics.

    Parameters
    ----------
    actives_ypredict : 1D array
        The predicted probabilities for active compounds.
    inactives_ypredict : 1D array
        The predicted probabilities for inactive compounds.

    Return
    ------
    weighted_accuracy : float 
    mcc : float
    roc : float
    prauc : float
    brier : float
    bedroc : float
    avg_prob_inactive : float 
        The averaged predicted probability of being active for inactive compounds.
    std_prob_inactive : float
        The standard deviation of predicted probability of being active for inactive compounds.
    avg_prob_active : float
        The averaged predicted probability of being active for active compounds.
    std_prob_active : float
        The standard deviation of predicted probability of being active for active compounds
    precision_inactive : float
    recall_inactive : float
    f1_inactive : float 
    precision_active : float 
    recall_active : float 
    f1_active: float 
    '''
    label = np.hstack((np.ones(len(actives_ypredict)),
                       np.zeros(len(inactives_ypredict)),
                       ))  # True labels
    sw = compute_sample_weight('balanced', label)
    probs = np.hstack((actives_ypredict, inactives_ypredict)
                      )  # Predicted probabilities
    pred = (probs > 0.5).astype(int)  # Predicted labels

    weighted_accuracy = accuracy_score(label, pred, sample_weight=sw)
    mcc = matthews_corrcoef(label, pred)
    roc = roc_auc_score(label, probs, average='weighted')
    prauc = average_precision_score(label, probs, average='weighted')
    brier = brier_score_loss(label, probs, sample_weight=sw)
    bedroc = CalcBEDROC(sorted(zip(probs, label), reverse=True), 1, 20)
    avg_prob_inactive = np.average(inactives_ypredict)
    std_prob_inactive = np.std(inactives_ypredict)
    avg_prob_active = np.average(actives_ypredict)
    std_prob_active = np.std(actives_ypredict)

    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    precision_active = tp/(tp+fp)
    recall_active = tp/(tp+fn)
    f1_active = 2*precision_active*recall_active / \
        (precision_active+recall_active)
    precision_inactive = tn/(tn+fn)
    recall_inactive = tn/(tn+fp)
    f1_inactive = 2*precision_inactive*recall_inactive / \
        (precision_inactive+recall_inactive)

    return (weighted_accuracy,
            mcc,
            roc,
            prauc,
            brier,
            bedroc,
            avg_prob_inactive,
            std_prob_inactive,
            avg_prob_active,
            std_prob_active,
            precision_inactive,
            recall_inactive,
            f1_inactive,
            precision_active,
            recall_active,
            f1_active
            )

def do_splitting(actives, inactives, seed, model_weights, groups=None):
    '''
    Split the dataset. Do cross validation. Return the metrics.

    Parameters
    ----------
    actives : 2D array
        Active compounds fingerprints.
    inactives : 2D array
        Inactive compounds fingerprints.
    seed : int
        random seed.
    model_weights : 
        Sequential model weights for each layer.
    groups : 1D array
        Group labels. If groups = None, then do TimeSeriesSplit.

    Return
    ------
    (avg_bedroc, std_bedroc,avg_roc,std_roc,avg_prauc,std_prauc, sizes)
        Apart from sizes which is string, others are all float.
    '''
    # Do splitting
    if groups is None:
        # Do time series splitting.
        tscv = TimeSeriesSplit(n_splits=5)
        a_split = [spl for spl in tscv.split(actives)]
        i_split = [spl for spl in tscv.split(inactives)]
    else:
        # Do group spliting.
        train_size = 0.5
        err = (None, None, None, None, None, None, "")  # Default empty return.

        # Check that there are at least 4 groups (enough across splits).
        n_splits = 4
        if not len(set(groups)) >= n_splits:
            return err

        gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size,
                                random_state=seed)

        # Check that we can split actives.
        try:
            a_split = [spl for spl in gss.split(
                actives, range(len(actives)), groups=groups)]
        except ValueError:
            return err

        # Check that there are at least 5 comps across splits.
        if all([len(splt[0]) >= 5 for splt in a_split]) == False:
            return err

        inact_ss = ShuffleSplit(n_splits=4, train_size=train_size,
                                random_state=seed)
        i_split = [spl for spl in inact_ss.split(inactives)]

    
    # Do cross-validatioin
    bedroc_list = []
    roc_list = []
    prauc_list = []
    for spl in zip(a_split, i_split):  # iterate across four splits
        a_train, a_test = spl[0][0], spl[0][1]
        i_train, i_test = spl[1][0], spl[1][1]
        x = np.vstack((actives[a_train], inactives[i_train]))
        y = [1] * len(a_train) + [0] * len(i_train)
        x = np.array(x, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)
        test_x = np.vstack((actives[a_test], inactives[i_test]))
        test_v = [1] * len(a_test) + [0] * len(i_test)
        test_x = np.array(test_x, dtype=np.uint8)
        test_v = np.array(test_v, dtype=np.uint8)
        clf = get_model(x[0].shape, model_weights)
        fit_model(clf, x, y)
        probs = clf.predict(test_x)
        tf.keras.backend.clear_session()
        prauc_list.append(average_precision_score(
            test_v, probs, average='weighted'))
        bedroc_list.append(CalcBEDROC(
            sorted(zip(probs, test_v), reverse=True), 1, 20))
        roc_list.append(roc_auc_score(test_v, probs, average='weighted'))

    avg_bedroc = np.average(bedroc_list)
    std_bedroc = np.std(bedroc_list)
    avg_roc = np.average(roc_list)
    std_roc = np.std(roc_list)
    avg_prauc = np.average(prauc_list)
    std_prauc = np.std(prauc_list)
    sizes = ','.join(
        map(str, [str(len(splt[0]))+'|'+str(len(splt[1])) for splt in a_split]))
    return (avg_bedroc, std_bedroc, avg_roc, std_roc, avg_prauc, std_prauc, sizes)

class MyClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(MyClassifierAdapter, self).__init__(model, fit_params)

    def fit(self, x, y):
        fit_model(self.model, x, y)

    def predict(self, x):
        c1 = np.array(self.model.predict(x)).reshape(-1, 1)
        c0 = 1-c1
        arr = np.hstack((c0, c1))
        return arr

class Train():
    def __init__(self, dir_inputs: str, dir_outputs: str):
        '''
        Parameters
        ----------
        dir_inputs : str
            Path to the folder that contain npz files which contain training inputs.
        dir_outputs : str
            Path to the folder for storing training outputs.
        '''
        # Directory of training inputs.
        self.inputs = dir_inputs
        # Directory to store the pkls for each model.
        self.pkls = os.path.join(dir_outputs, 'ANN_pkls/')
        # Directory to store the training log.
        self.outputs = dir_outputs

        self.seed = 123  # Random seed.
        self.dic_compounds = {} # The dictionary of compounds for the current model.
        self.log = Log() # The log of the current model.
        self.mid = '' # The model_id of the current model.

        if os.path.isdir(dir_outputs):
            rmtree(dir_outputs)
        os.mkdir(dir_outputs)
        os.mkdir(self.pkls)
        return

    def train(self):
        count = 0
        total = len(os.listdir(self.inputs))
        # Train for each model in the training-inputs folder.
        for f in os.listdir(self.inputs):
            # Retrieve data for the model.
            # actives contains active compounds. It is a 2D array. Each row is the 2048 bit molecular fingerprint.
            # inactives contains inactive compounds. It is also a 2D array.
            # pids is a 1D array, which contains the publication_id for each active compound.
            # scafs is a 1D array, which contains the scaffold_id for each active compound.
            mid_data = np.load(os.path.join(self.inputs, f))
            actives, inactives, pids, scafs = mid_data['actives'], mid_data[
                'inactives'], mid_data['pids'], mid_data['scafs']

            # Get the model_id for each model.
            self.mid = f.split('.')[0]
            print(self.mid + ' modelling has started.')

            # log data for the model
            self.log = Log(self.mid, 'ECFP_4', len(set(scafs)), len(set(pids)))

            # Do train_dev_test split
            self.dic_compounds, enough_compounds = train_dev_test_split(
                actives,
                pids,
                scafs,
                inactives,
                test_size=0.25,
                dev_size=0.25,
                min_size=10,
                random_seed=self.seed)

            if enough_compounds:
                self.calculate()

            # Write training log
            write_training_log(self.log, self.outputs)
            count = count + 1
            print('%d models have finished, %d models in total' %
                (count, total))
        return
    
    def calculate(self):
        # Get and fit the classifier. Time the process.
        clf = get_model(self.dic_compounds['actives_train'][0].shape)
        x = np.vstack(
            (self.dic_compounds['actives_train'], self.dic_compounds['inactives_train']))
        y = np.vstack(
            (self.dic_compounds['actives_train_y'], self.dic_compounds['inactives_train_y']))
        start_time = time.time()
        fit_model(clf, x, y)
        self.log.train_time_sec = time.time() - start_time

        # Get the predicted probabilities.
        aprobs_train = clf.predict(self.dic_compounds['actives_train']).ravel()
        iprobs_train = clf.predict(
            self.dic_compounds['inactives_train']).ravel()
        aprobs_test = clf.predict(self.dic_compounds['actives_test']).ravel()
        iprobs_test = clf.predict(self.dic_compounds['inactives_test']).ravel()

        # Get the performance metrics.
        # train set performance
        (self.log.train_weighted_accuracy,
            self.log.train_mcc,
            self.log.train_roc,
            self.log.train_prauc,
            self.log.train_brier,
            self.log.train_bedroc,
            self.log.train_avg_prob_inactive,
            self.log.train_std_prob_inactive,
            self.log.train_avg_prob_active,
            self.log.train_std_prob_active,
            self.log.train_precision_inactive,
            self.log.train_recall_inactive,
            self.log.train_f1_inactive,
            self.log.train_precision_active,
            self.log.train_recall_active,
            self.log.train_f1_active
            ) = calculate_performance(aprobs_train, iprobs_train)
        # test set performance
        (self.log.test_weighted_accuracy,
            self.log.test_mcc,
            self.log.test_roc,
            self.log.test_prauc,
            self.log.test_brier,
            self.log.test_bedroc,
            self.log.test_avg_prob_inactive,
            self.log.test_std_prob_inactive,
            self.log.test_avg_prob_active,
            self.log.test_std_prob_active,
            self.log.test_precision_inactive,
            self.log.test_recall_inactive,
            self.log.test_f1_inactive,
            self.log.test_precision_active,
            self.log.test_recall_active,
            self.log.test_f1_active
            ) = calculate_performance(aprobs_test, iprobs_test)

        # Get the weights of the model so that we do not need to retrain later.
        weights = clf.get_weights()

        # Save model.
        clf.save(os.path.join(self.pkls, self.mid + '.h5'))

        # Clear session, otherwise memory will be used up.
        tf.keras.backend.clear_session()

        # LEAVE 50% OF PUBS OUT
        (self.log.pub_avg_bedroc,
            self.log.pub_std_bedroc,
            self.log.pub_avg_roc,
            self.log.pub_std_roc,
            self.log.pub_avg_prauc,
            self.log.pub_std_prauc,
            self.log.pub_size) = do_splitting(self.dic_compounds['actives_train'],
                                        self.dic_compounds['inactives_train'],
                                        self.seed,
                                        weights,
                                        groups=self.dic_compounds['pids_train'])
        # LEAVE 50% OF SCAFFOLDS OUT
        (self.log.scaf_avg_bedroc,
            self.log.scaf_std_bedroc,
            self.log.scaf_avg_roc,
            self.log.scaf_std_roc,
            self.log.scaf_avg_prauc,
            self.log.scaf_std_prauc,
            self.log.scaf_size) = do_splitting(self.dic_compounds['actives_train'],
                                        self.dic_compounds['inactives_train'],
                                        self.seed,
                                        weights,
                                        groups=self.dic_compounds['scafs_train'])
        # Time Series split
        (self.log.tscv_avg_bedroc,
            self.log.tscv_std_bedroc,
            self.log.tscv_avg_roc,
            self.log.tscv_std_roc,
            self.log.tscv_avg_prauc,
            self.log.tscv_std_prauc,
            self.log.tscv_size) = do_splitting(self.dic_compounds['actives_train'],
                                        self.dic_compounds['inactives_train'],
                                        self.seed,
                                        weights)

        # Conformal Prediction
        (self.log.one_class_rate, 
        self.log.true_active_rate, 
        self.log.true_inactive_rate) = self.conformal_prediction()
        return

    def conformal_prediction(self):
        '''
        Return
        ------
        one_class_rate : float 
            The proportion of compounds that are assigned to 
            one class only (either active or inactive).
        true_active_rate : float
            The proportion of compounds that not only 
            are predicted to be active by the conformal prediction, 
            but also have correct labels.
        true_inactive_rate : float 
            The proportion of compounds that not only are predicted to be 
            inactive by the conformal prediction, but also have correct labels.
        '''
        clf = get_model(self.dic_compounds['actives_train'][0].shape)
        nc = ClassifierNc(MyClassifierAdapter(clf))
        icp = IcpClassifier(nc)  # Create an inductive conformal classifier

        # Fit the ICP using the training set
        x = np.vstack((self.dic_compounds['actives_train'],
                    self.dic_compounds['inactives_train']))
        y = np.vstack((self.dic_compounds['actives_train_y'],
                    self.dic_compounds['inactives_train_y']))
        icp.fit(x, y)

        # Calibrate the ICP using the calibration set
        x = np.vstack((self.dic_compounds['actives_dev'],
                    self.dic_compounds['inactives_dev']))
        y = np.vstack((self.dic_compounds['actives_dev_y'],
                    self.dic_compounds['inactives_dev_y']))
        icp.calibrate(x, y)

        # Produce predictions for the test set, with confidence 95%
        x = np.vstack((self.dic_compounds['actives_test'],
                    self.dic_compounds['inactives_test']))
        prediction = icp.predict(x, significance=0.05)

        # Matrix : 2D Dataframe
        # One compound per row.
        # The first column is 1, if the 
        # compound is predicted to be inactive with in 95% CI.
        # The second column is 1, if the 
        # compound is predicted to be active within 95 % CI.
        # The third column has the true labels for each compound. 1 for active
        # compounds, 0 for inactive compounds.
        y = np.vstack((self.dic_compounds['actives_test_y'],
                    self.dic_compounds['inactives_test_y']))
        matrix = pd.DataFrame(np.hstack((prediction, y)))

        # Find the proportion of the compounds that are assigned to one class only.
        fun = lambda row :  (row[0]+row[1]) == 1
        arr = matrix.apply(fun, axis=1)
        one_class_rate = sum(arr)/len(matrix)

        eps = 1e-10  # Avoid 0/0

        # For the compounds that conformal prediction predicts to be in the active class only,
        # find the proportion of them which are correctly predicted by the model.
        fun = lambda row : ((row[0] == 0) & (row[1] == 1) & (row[2] == 1))
        arr1 = matrix.apply(fun, axis=1)
        fun = lambda row : ((row[0] == 0) & (row[1] == 1))
        arr2 = matrix.apply(fun, axis=1)
        true_active_rate = sum(arr1)/(sum(arr2)+eps)

        # For the compounds that conformal prediction predicts to be in the inactive class only,
        # find the proportion of them which are correctly predicted by the model.
        fun = lambda row : ((row[0] == 1) & (row[1] == 0) & (row[2] == 0))
        arr1 = matrix.apply(fun, axis=1)
        fun = lambda row : ((row[0] == 1) & (row[1] == 0))
        arr2 = matrix.apply(fun, axis=1)
        true_inactive_rate = sum(arr1)/(eps+sum(arr2))

        return (one_class_rate, true_active_rate, true_inactive_rate)        


