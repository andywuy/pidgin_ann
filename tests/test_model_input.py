import pidgin_ann
import numpy as np


def test_model_input():
    m1 = pidgin_ann.Model("tests/test_data/datasets/",
                          "tests/test_data/classes_info.txt",
                          "model_inputs", n=1)
    m1.create()
    output = np.load('model_inputs/P11309,SE_100.npz')
    expected = np.load('tests/test_data/npz/P11309,SE_100.npz')
    check_pid = np.all(expected['pids'] == output['pids'])
    assert check_pid==True
