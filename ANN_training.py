### Train the artificial neural network.
### The trained model is saved in .h5 files. The training results are saved in the training log.
### The code is adapted from the original PIDGINv4 code.

# libraries
import time
import itertools
import os
import numpy as np
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score # for the prauc score
from sklearn.metrics import roc_auc_score # for the roc score
from sklearn.metrics import brier_score_loss,accuracy_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupShuffleSplit, ShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel('ERROR') # suppress warning from tf

from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.cp import IcpClassifier

def _RIEHelper(scores, col, alpha):
    '''
    modify from rdkit scoring
    '''
    numMol = len(scores)
    alpha = float(alpha)
    if numMol == 0:
        raise ValueError('score list is empty')
    if alpha <= 0.0:
        raise ValueError('alpha must be greater than zero')

    denom = 1.0 / numMol * ((1 - math.exp(-alpha)) /
                            (math.exp(alpha / numMol) - 1))
    numActives = 0
    sum_exp = 0

    # loop over score list
    for i in range(numMol):
        active = scores[i][col]
        if active:
            numActives += 1
            sum_exp += math.exp(-(alpha * (i + 1)) / numMol)

    if numActives > 0:  # check that there are actives
        RIE = sum_exp / (numActives * denom)
    else:
        RIE = 0.0

    return RIE, numActives


def CalcBEDROC(scores, col, alpha):
    """ 
    modify from rdkit scoring
    BEDROC original defined here:
      Truchon, J. & Bayly, C.I.
      Evaluating Virtual Screening Methods: Good and Bad Metric for the "Early Recognition"
      Problem. J. Chem. Inf. Model. 47, 488-508 (2007).
      ** Arguments**

        - scores: 2d list or numpy array
               0th index representing sample
               scores must be in sorted order with low indexes "better"
               scores[sample_id] = vector of sample data
        -  col: int
               Index of sample data which reflects true label of a sample
               scores[sample_id][col] = True iff that sample is active
        -  alpha: float
               hyper parameter from the initial paper for how much to enrich the top
       **Returns**
         float BedROC score
      """
    # calculate RIE
    RIE, numActives = _RIEHelper(scores, col, alpha)

    if numActives > 0:
        numMol = len(scores)
        ratio = 1.0 * numActives / numMol
        RIEmax = (1 - math.exp(-alpha * ratio)) / \
            (ratio * (1 - math.exp(-alpha)))
        RIEmin = (1 - math.exp(alpha * ratio)) / \
            (ratio * (1 - math.exp(alpha)))

        if RIEmax != RIEmin:
            BEDROC = (RIE - RIEmin) / (RIEmax - RIEmin)
        else:  # numActives = numMol
            BEDROC = 1.0
    else:
        BEDROC = 0.0

    return BEDROC

def do_group_splitting(actives,groups,inactives,dic_constants,model_weights):
    train_size=0.5
    err = ('','','','','')
    #check 4 groups (enough across splits)
    if not len(set(groups)) >= 4: return err
    gss = GroupShuffleSplit(n_splits=4, train_size=train_size,
                            random_state=dic_constants['random_seed'])
    #check can split actives
    try: a_split = [spl for spl in gss.split(actives, range(len(actives)), groups=groups)]
    except ValueError: return err
    #check at least 5 comps acrross splits
    if all([len(splt[0])>=5 for splt in a_split]) == False: return err
    inact_ss = ShuffleSplit(n_splits=4, train_size=train_size,
                            random_state=dic_constants['random_seed'])
    i_split = [spl for spl in inact_ss.split(inactives)]
    bedroc = []
    rocs = []
    prauc = []
    bs = []
    for spl in zip(a_split,i_split):# iterate across four splits
        a_train, a_test = spl[0][0], spl[0][1]
        i_train, i_test = spl[1][0], spl[1][1]
        x = np.vstack((actives[a_train],inactives[i_train]))
        y = [1] * len(a_train) + [0] * len(i_train)
        x = np.array(x,dtype=np.uint8)
        y = np.array(y,dtype=np.uint8)        
        test_x = np.vstack((actives[a_test],inactives[i_test]))
        test_v = [1] * len(a_test) + [0] * len(i_test)
        test_x = np.array(test_x,dtype=np.uint8)
        test_v = np.array(test_v,dtype=np.uint8)
        sw = compute_sample_weight('balanced',y)
        model = get_fit_model(x,y,sw,model_weights)
        probs = model.predict(test_x)
        tf.keras.backend.clear_session()
        prauc.append(average_precision_score(test_v,probs,average='weighted'))
        pinp = sorted(zip(probs,test_v), reverse=True)
        bedroc.append(CalcBEDROC(pinp,1,20))
        rocs.append(roc_auc_score(test_v,probs,average='weighted'))
        test_sw = compute_sample_weight('balanced',test_v)
        bs.append(brier_score_loss(test_v,probs,sample_weight=test_sw))
    bedroc_ret_gss = ','.join(map(str,["%.3f" %f for f in [np.average(bedroc), np.median(bedroc), np.std(bedroc)]]))
    roc_ret_gss = ','.join(map(str,["%.3f" %f for f in [np.average(rocs), np.median(rocs), np.std(rocs)]]))
    pr_auc_ret_gss = ','.join(map(str,["%.3f" %f for f in [np.average(prauc), np.median(prauc), np.std(prauc)]]))
    bs_ret_gss = ','.join(map(str,["%.3f" %f for f in [np.average(bs), np.median(bs), np.std(bs)]]))
    sizes = ','.join(map(str,[str(len(splt[0]))+'|'+str(len(splt[1])) for splt in a_split]))
    return bedroc_ret_gss,roc_ret_gss,pr_auc_ret_gss,bs_ret_gss,sizes

def do_tscv(actives,inactives,model_weights):
    tscv = TimeSeriesSplit(n_splits=5)
    a_split = [spl for spl in tscv.split(actives)]
    i_split = [spl for spl in tscv.split(inactives)]
    sizes = ','.join(map(str,[str(len(splt[0]))+'|'+str(len(splt[1])) for splt in a_split]))
    bedroc = []
    rocs = []
    prauc = []
    bs = []
    for spl in zip(a_split,i_split):
        a_train, a_test = spl[0][0], spl[0][1]
        i_train, i_test = spl[1][0], spl[1][1]
        x = np.vstack((actives[a_train],inactives[i_train]))
        y = [1] * len(a_train) + [0] * len(i_train)
        x = np.array(x,dtype=np.uint8)
        y = np.array(y,dtype=np.uint8)        
        test_x = np.vstack((actives[a_test],inactives[i_test]))
        test_v = [1] * len(a_test) + [0] * len(i_test)
        test_x = np.array(test_x,dtype=np.uint8)
        test_v = np.array(test_v,dtype=np.uint8)
        sw = compute_sample_weight('balanced',y)
        model =get_fit_model(x,y,sw,model_weights)
        probs = model.predict(test_x)
        tf.keras.backend.clear_session()
        prauc.append(average_precision_score(test_v,probs,average='weighted'))
        pinp = sorted(zip(probs,test_v), reverse=True)
        bedroc.append(CalcBEDROC(pinp,1,20))
        rocs.append(roc_auc_score(test_v,probs,average='weighted'))
        test_sw = compute_sample_weight('balanced',test_v)
        bs.append(brier_score_loss(test_v,probs,sample_weight=test_sw))
    bedroc_ret = ','.join(map(str,["%.3f" %f for f in [np.average(bedroc), np.median(bedroc), np.std(bedroc)]]))
    roc_ret = ','.join(map(str,["%.3f" %f for f in [np.average(rocs), np.median(rocs), np.std(rocs)]]))
    pr_auc_ret = ','.join(map(str,["%.3f" %f for f in [np.average(prauc), np.median(prauc), np.std(prauc)]]))
    bs_ret = ','.join(map(str,["%.3f" %f for f in [np.average(bs), np.median(bs), np.std(bs)]]))
    return bedroc_ret,roc_ret,pr_auc_ret,bs_ret,sizes

def grid_search_model(dic_compounds):
    
    def gen_model(x_train, y_train, params_input):
        sw = compute_sample_weight('balanced',y_train)
        model = Sequential()
        model.add(tf.keras.Input(shape=x_train[0].shape))
        # params_input[0] number of layers
        # params_input[1] number of hidden units in each layer
        for i in range(params_input[0]):
            model.add(Dense(params_input[1],activation='relu'))
        
        model.add(Dense(1,activation='sigmoid'))
        opt = Adam(learning_rate=0.001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)    
        model.compile(optimizer=opt,loss='binary_crossentropy',
                      weighted_metrics=[tf.keras.metrics.BinaryAccuracy()])
        model.fit(x_train,y_train,
                            sample_weight=sw,batch_size=128,epochs=100,
              verbose=0,callbacks=[early_stopping])
        return model  

    params = {'num_layer':[1,3,5],
              'num_neuron':[10,100,1000],
              }
    params_inputs = list(itertools.product(params['num_layer'],
                            params['num_neuron']))
    
    x_train = np.concatenate((dic_compounds['actives_train'],
                              dic_compounds['inactives_train']))
    y_train = np.concatenate((np.ones(len(dic_compounds['actives_train'])),
                              np.zeros(len(dic_compounds['inactives_train'])))) 
    
    results = []
    for i in params_inputs:
        model = gen_model(x_train, y_train,i)
        aprobs_dev = model.predict(dic_compounds['actives_dev']).ravel()
        iprobs_dev = model.predict(dic_compounds['inactives_dev']).ravel()
        tf.keras.backend.clear_session() #otherwise memory will be used up
        # dev set performance
        metrics = calculate_performance(aprobs_dev,iprobs_dev)
        lst = list(i)+ list(metrics)
        results.append(lst)
    
        
    columns = ['num_layer','num_neuron']    
    # copy from calculate performance function
    str1 = '''roc, prauc, brier, bedroc, pc_i, pc_a, 
             weighted_accuracy,mcc,
             actives_precision,actives_recall,actives_f1,
             inactives_precision,inactives_recall,inactives_f1'''
    for s in str1.split(','):  
        columns.append('dev_'+s.strip())
        
    df = pd.DataFrame(results,columns=columns)
    
    return df

def get_fit_model(matrix,label,sample_weight,model_weights=None):
    #reg = tf.keras.regularizers.l2(0.01)
    model = Sequential([Dense(100,input_shape=matrix[0].shape,activation='relu'),
                        BatchNormalization(),
                        Dense(10,activation='relu'),
                        Dense(1,activation='sigmoid')])
    if model_weights != None: model.set_weights(model_weights)
    opt = Adam(learning_rate=0.001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=opt,loss='binary_crossentropy')
    model.fit(matrix,label,sample_weight=sample_weight,batch_size=128,epochs=100,
              verbose=0,callbacks=[early_stopping])
    return model

def calculate_performance(actives_ypredict,inactives_ypredict):    
    label = np.hstack((np.ones(len(actives_ypredict)),
                       np.zeros(len(inactives_ypredict)),
                       ))
    sw = compute_sample_weight('balanced',label)
    probs = np.hstack((actives_ypredict,inactives_ypredict))
    pred = (probs>0.5).astype(int)
    roc = round(roc_auc_score(label,probs,average='weighted'),3)
    prauc = round(average_precision_score(label,probs,average='weighted'),3)
    sw = compute_sample_weight('balanced',label)
    brier = round(brier_score_loss(label, probs,sample_weight=sw),3)
    pinp = sorted(zip(probs,label), reverse=True)
    bedroc = round(CalcBEDROC(pinp,1,20),3)
    pc_i = str(round(np.average(inactives_ypredict),3)) + ',' + str(round(np.std(inactives_ypredict),3))
    pc_a = str(round(np.average(actives_ypredict),3)) + ',' + str(round(np.std(actives_ypredict),3))    
    #return roc,prauc,bedroc,brier,pc_i,pc_a

    weighted_accuracy = round(accuracy_score(label,pred,sample_weight=sw),3)
    mcc = round(matthews_corrcoef(label,pred),3)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    actives_precision = round(tp/(tp+fp),3)
    actives_recall = round(tp/(tp+fn),3)
    actives_f1 = round(2*actives_precision*actives_recall/(actives_precision+actives_recall),3)
    inactives_precision = round(tn/(tn+fn),3)
    inactives_recall = round(tn/(tn+fp),3)
    inactives_f1 = round(2*inactives_precision*inactives_recall/(inactives_precision+inactives_recall), 3)
    
    metrics=(roc, prauc, brier, bedroc, pc_i, pc_a, 
             weighted_accuracy,mcc,
             actives_precision,actives_recall,actives_f1,
             inactives_precision,inactives_recall,inactives_f1,)
    return metrics
def evaluate_conformal_prediction(matrix):
    matrix = pd.DataFrame(matrix)
    # the first column is for inactive compounds class 0
    # if it is 1, then the class is with in 95% confidence region
    # the second column is for active compounds class 1
    # the third column is for true compound label
    # the fourth column is the predicted probability for inactivity
    # the fifth column is the predicted probability for activity
    # the sixth column is the predicted label
    
    # measure the efficiency
    fun = lambda row: (row[0]+row[1])==1
    arr = matrix.apply(fun,axis=1)
    conf_one_class_rate =round( sum(arr)/len(matrix),3)
    
    eps =1e-10
    # for conformal prediction that predicts the active (1) class only,
    # what is the percentage of correct prediction
    fun = lambda row: ((row[0]==0) & (row[1]==1) & (row[2]==1))
    arr1 = matrix.apply(fun,axis=1)
    fun = lambda row: ((row[0]==0) & (row[1]==1 ))
    arr2 =  matrix.apply(fun,axis=1)
    conf_true_active_rate = round(sum(arr1)/(sum(arr2)+eps),3)
    # for conformal prediction that predicts the inactive (1) class only,
    # what is the percentage of correct prediction    
    fun = lambda row: ((row[0]==1) & (row[1]==0) & (row[2]==0))
    arr1 = matrix.apply(fun,axis=1)
    fun = lambda row: ((row[0]==1) & (row[1]==0) )
    arr2 =  matrix.apply(fun,axis=1)
    conf_true_inactive_rate = round( sum(arr1)/(eps+sum(arr2)),3)    
    results = [conf_one_class_rate,conf_true_active_rate,conf_true_inactive_rate]
    return results

def write_compounds_results(pred_results,mlabel,model_name,path):
    compound_id = np.arange(len(pred_results)).reshape(-1,1)
    model_id = (np.ones(len(pred_results))*mlabel).astype(np.int).reshape(-1,1)    
    arr = np.hstack((model_id,
                          compound_id,
                          pred_results))
    columns=['model_id',
            'compound_id',
            '?conf_inactive',
            '?conf_active',
            'true_labels',
            'iprob',
            'aprob',
            'pred_labels']
    df = pd.DataFrame(arr,columns=columns)
    df.to_csv(path+model_name+'compound_prediction_results.txt',
                       sep='\t',
                       index=False)
    
    return 

def train_dev_test_split(actives,pids,scafs,inactives,test_size,dev_size,min_size,random_seed):
    '''
    split the data
    test_size,dev_size must be decimal e.g. 0.25
    min_size is the minimum number of compounds in each set
    '''
    import math
    
    enough_compounds = True
    dic_compounds = {}
    np.random.seed(random_seed)
    #for active compounds
    idx = np.random.permutation(len(actives))
    num_test = math.floor(len(actives)*test_size)
    num_dev = math.floor(len(actives)*dev_size)
    test_idx = idx[0:num_test]
    dev_idx = idx[num_test:num_dev+num_test]
    train_idx = idx[num_dev+num_test:]
    if len(test_idx)<min_size | len(dev_idx)<min_size | len(train_idx)<min_size :
        enough_compounds = False        
    else:
        dic_compounds['pids_train'] = pids[train_idx]
        dic_compounds['scafs_train'] = scafs[train_idx]
        dic_compounds['actives_train']= actives[train_idx]
        dic_compounds['actives_train_y']=np.ones(len(train_idx)).reshape(-1,1) #2d vector
        dic_compounds['pids_dev']=pids[dev_idx]
        dic_compounds['scafs_dev']=scafs[dev_idx]
        dic_compounds['actives_dev'] = actives[dev_idx]
        dic_compounds['actives_dev_y']=np.ones(len(dev_idx)).reshape(-1,1)
        dic_compounds['actives_test'] = actives[test_idx]
        dic_compounds['actives_test_y']=np.ones(len(test_idx)).reshape(-1,1)
        
    #for inactive compounds
    idx = np.random.permutation(len(inactives))
    num_test = math.floor(len(inactives)*test_size)
    num_dev = math.floor(len(inactives)*dev_size)
    test_idx = idx[0:num_test]
    dev_idx = idx[num_test:num_dev+num_test]
    train_idx = idx[num_dev+num_test:]
    if len(test_idx)<min_size | len(dev_idx)<min_size | len(train_idx)<min_size :
        enough_compounds = False        
    else:
        dic_compounds['inactives_train'] = inactives[train_idx]
        dic_compounds['inactives_train_y'] = np.zeros(len(train_idx)).reshape(-1,1)
        dic_compounds['inactives_dev'] = inactives[dev_idx]
        dic_compounds['inactives_dev_y'] = np.zeros(len(dev_idx)).reshape(-1,1)
        dic_compounds['inactives_test'] = inactives[test_idx]  
        dic_compounds['inactives_test_y'] = np.zeros(len(test_idx)).reshape(-1,1)
  
    return dic_compounds,enough_compounds

class MyClassifierAdapter(ClassifierAdapter):
        def __init__(self, model, fit_params=None):
            super(MyClassifierAdapter, self).__init__(model, fit_params)

        def fit(self,x,y):
            fit_model(self.model,x,y)
        
        def predict(self,x):
            arr = predict_model(self.model,x)
            return arr
        
def get_model(input_shape,model_weights=None):
    model = Sequential([Dense(100,input_shape=input_shape,activation='relu'),
                        BatchNormalization(),
                        Dense(10,activation='relu'),
                        Dense(1,activation='sigmoid')])
    if model_weights != None: model.set_weights(model_weights)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss='binary_crossentropy')
    return model

def fit_model(model,x,y):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    sw = compute_sample_weight('balanced',y)
    model.fit(x,y,sample_weight=sw,
              batch_size=128,epochs=100,
              verbose=0,callbacks=[early_stopping])
    return 

def predict_model(model,x):
    c1 = np.array(model.predict(x)).reshape(-1,1)
    c0 = 1-c1
    arr = np.hstack((c0,c1))
    return arr
      
def conformal_prediction(dic_compounds):
    clf = get_model(dic_compounds['actives_train'][0].shape)
    nc = ClassifierNc(MyClassifierAdapter(clf))
    icp = IcpClassifier(nc)	# Create an inductive conformal classifier
    
    # Fit the ICP using the proper training set
    x= np.vstack((dic_compounds['actives_train'],dic_compounds['inactives_train']))
    y = np.vstack((dic_compounds['actives_train_y'],dic_compounds['inactives_train_y']))
    icp.fit(x,y)
    
    # Calibrate the ICP using the calibration set
    x= np.vstack((dic_compounds['actives_dev'],dic_compounds['inactives_dev']))
    y = np.vstack((dic_compounds['actives_dev_y'],dic_compounds['inactives_dev_y']))
    icp.calibrate(x, y)
    
    # Produce predictions for the test set, with confidence 95%
    x = np.vstack((dic_compounds['actives_test'],dic_compounds['inactives_test']))
    prediction = icp.predict(x, significance=0.05)
    
    # the first column is for inactive compounds class 0
    # if it is 1, then the class is with in 95% confidence region
    # the second column is for active compounds class 1
    # the third column is for true compound label
    # the fourth column is the predicted probability for inactivity
    # the fifth column is the predicted probability for activity
    # the sixth column is the predicted label
    y = np.vstack((dic_compounds['actives_test_y'],dic_compounds['inactives_test_y']))
    clf = icp.nc_function.model.model
    probs = predict_model(clf, x)
    pred_labels = ((probs[:,1])>0.5).astype(int).reshape(-1,1)
    matrix = np.hstack((prediction,y,probs,pred_labels))
    return matrix,clf
    
def do_modelling(actives, pids, scafs, inactives, model_name, model_label, dic_constants):
    dic_compounds, enough_compounds = train_dev_test_split(actives, pids, scafs, inactives,
                         test_size=0.25, dev_size=0.25,
                         min_size=10,
                         random_seed=dic_constants['random_seed']) 
        
    if enough_compounds:        
        # conformal prediction. Use cal as train set
        start_time = time.time()
        pred_results, model = conformal_prediction(dic_compounds)
        elapsed_time = time.time() - start_time
        aprobs_train = model.predict(dic_compounds['actives_train']).ravel()
        iprobs_train = model.predict(dic_compounds['inactives_train']).ravel()
        aprobs_test = model.predict(dic_compounds['actives_test']).ravel()
        iprobs_test = model.predict(dic_compounds['inactives_test']).ravel()
        weights = model.get_weights()
        # save model
        model.save(dic_constants['dir_model_pkls'] + model_name + '.h5')
        tf.keras.backend.clear_session() #otherwise memory will be used up
        
        # write compounds prediction results
        write_compounds_results(pred_results,
                                mlabel,model_name,
                                dic_constants['dir_model_results'])

        # train set performance
        results_train = calculate_performance(aprobs_train,iprobs_train)
        # test set performance
        results_test = calculate_performance(aprobs_test,iprobs_test)
        # conformal prediction metrics
        results_conf = evaluate_conformal_prediction(pred_results)
        
        # for cross validation, combine dev and train
        actives_train = np.vstack((dic_compounds['actives_train'],
                            dic_compounds['actives_dev']))
        inactives_train = np.vstack((dic_compounds['inactives_train'],
                            dic_compounds['inactives_dev']))
        
        ##LEAVE 50% OF PUBS OUT
        results_pss = do_group_splitting(actives_train,
                                        np.concatenate([dic_compounds['pids_train'],dic_compounds['pids_dev']]),
                                        inactives_train,
                                        dic_constants,
                                        weights)
        ##LEAVE 50% OF SCAFFOLDS OUT
        results_sss = do_group_splitting(actives_train,
                                        np.concatenate([dic_compounds['scafs_train'],dic_compounds['scafs_dev']]),
                                        inactives_train,
                                        dic_constants,
                                        weights)
        ###------TSCV------###
        results_tscv = do_tscv(actives_train,inactives_train,weights)
        ret= [model_name, 'ECFP_4', len(set(scafs)), len(set(pids)),
              round(elapsed_time,2)]
        lst = [results_test, results_train, results_pss,
                   results_sss, results_tscv, results_conf]
        for l in lst:
            ret.extend(l)

    else:
        ret = [model_name, 'ECFP_4', len(set(scafs)), len(set(pids))]

    return ret

def get_models_list(path,start,end):
    '''
    get the list of all models used in the training
    path is the directory of models
    select models from start to end in all models
    the mids_list contains the unique model number and the model name
    '''
    mids_list=[]
    with open(path + "mids_list.txt", "r") as f:
        for count, line in enumerate(f):
            mids_list.append((count,line.strip()))
        f.close()  
    
    mids_list = mids_list[start:end]
    return mids_list


def write_training_log(logs,path):
    fname = 'training_log.txt' 
    logf = open(path+fname,'w')
    # copy from calculate performance function
    str1 = '''roc, prauc, brier, bedroc, pc_i, pc_a, 
             weighted_accuracy,mcc,
             actives_precision,actives_recall,actives_f1,
             inactives_precision,inactives_recall,inactives_f1'''
       
    #copy from do group splitting function
    str2 =''' bedroc_ret,roc_ret,pr_auc_ret,bs_ret,sizes '''
    
    # copy from evaluate conformal prediction
    str3 = ''' conf_one_class_rate,conf_true_active_rate,conf_true_inactive_rate'''
    
    columns =['MODEL_ID','FINGERPRINT','N_SCAFFOLDS','N_PUBLICATIONS','TRAIN_TIME_SEC']
    for s in str1.split(','):
        columns.append('test_'+s.strip())
        
    for s in str1.split(','):
        columns.append('train_'+s.strip())
        
    for s in str2.split(','):
        columns.append('pub_'+s.strip())
        
    for s in str2.split(','):
        columns.append('scaf_'+s.strip())
        
    for s in str2.split(','):
        columns.append('tscv_'+s.strip())
    
    for s in str3.split(','):
        columns.append(s.strip())
        
    string = '\t'.join(columns)
    logf.write(string+'\n')
    for result in logs:
        logf.write('\t'.join(map(str,result)) + '\n')
    logf.close()
    return 

def initialisation():
    """
    initialise the require parameters, including:
    - the range of models used to train
    - the directory of the input model data
    - the directory of the output training models (pkls)
    - the directory of the output training results (the prediction for each compounds)
    - the directory of the output training log (different metrics)
    - the directory of the output parameter search for each model
    """
    start_id, end_id = 0, 1000
    random_seed = 123
    
    dir_model_inputs = "./model_inputs/"
    
    dir_model_pkls = './ANN_pkls_%dto%d/' % (start_id,end_id-1)
    if not os.path.exists(dir_model_pkls):
        os.mkdir(dir_model_pkls)
    
    dir_model_results = './ANN_results_%dto%d/' % (start_id,end_id-1)
    if not os.path.exists(dir_model_results):
        os.mkdir(dir_model_results)
        
    dir_training_log = './ANN_training_log_%dto%d/' % (start_id,end_id-1)
    if not os.path.exists(dir_training_log):
        os.mkdir(dir_training_log)
    
    dir_model_para = './ANN_parameter_search_%dto%d/' % (start_id,end_id-1)
    if not os.path.exists(dir_model_para):
        os.mkdir(dir_model_para)    
        
    dic =  {'start_id':start_id,
            'end_id':end_id,
            'random_seed':random_seed,
            'dir_model_inputs':dir_model_inputs,
            'dir_model_pkls':dir_model_pkls,
            'dir_model_results':dir_model_results,
            'dir_training_log':dir_training_log,
            'dir_model_para':dir_model_para,}
    return dic
 
if __name__ == "__main__":
    dic_constants = initialisation()
    mids_list = get_models_list(dic_constants['dir_model_inputs'],
                                dic_constants['start_id'],
                                dic_constants['end_id'])
    logs = [] #training log--containing measuring metrics for each model
    count = 0
    for mlabel, mid in mids_list:
        count = count + 1
        path = dic_constants['dir_model_inputs'] + mid + '.npz'
        mid_data = np.load(path)
        print(mid + ' modelling has started')
        actives, inactives, pids, scafs = mid_data['actives'], mid_data['inactives'], mid_data['pids'], mid_data['scafs']
        log = do_modelling(actives, pids, scafs, inactives, mid, mlabel, dic_constants)
        logs.append(log)
        print('%d models have finished, %d models in total' % (count, len(mids_list)))
        
    write_training_log(logs,dic_constants['dir_training_log'])
    print('Everything is done!')
