import pidgin_ann
import pandas as pd
import numpy as np

def test_train():
    m2 = pidgin_ann.Train("tests/test_data/npz", "training_outputs")
    m2.train()
    expected = pd.read_table('tests/test_data/training_log.txt', sep = '\t')
    output = pd.read_table('training_outputs/training_log.txt', sep = '\t')
    diff  = expected["test_weighted_accuracy"].values - output["test_weighted_accuracy"].values
    err = 0.1
    assert np.all(np.abs(diff) < err) == True