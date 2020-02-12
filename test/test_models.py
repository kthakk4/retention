import pytest
from retention import models,utils

testobj = models.ShiftedBetaGeom()

data_junk_vals = [
    ['one','two',3,4],[8,4,3,-1], [0,0,0,0],[]
]

@pytest.mark.parametrize("a",data_junk_vals)
def test_data_loading_bad_data(a):
    with pytest.raises(ValueError):
        testobj.load_training_data(a)

param_values = [
    (1,40,20),
    (20,40,20),
    (3,2,3),
    (3,0.1293,3),
    (4,3,0.2038),
    (0,0,0),
    (4,102938504309348,0),
    (3,3,1029833)
]

@pytest.mark.parametrize("t,a,b",param_values)
def test_get_churn_prob_t(t,a,b):
    p = testobj.get_churn_prob_t(t,a,b)
    assert (p >= 0.0) & (p<=1.0) #probability can' tbe outside of [0,1]
    with pytest.raises(AssertionError):
        testobj.get_churn_prob_t(-1,10,12)
        testobj.get_churn_prob_t(1000,2,3)


good_training_data = [
    [100,98,94,93,91,90],
    [100,98,95,92,91],
    [1,1,1,1,1],
    [100,50,50,50,50]
]

bad_training_data = [
    [1000,494,201,100],
    [100,101,98,94,93,91,90],
    [-100,-98,-97],
    [0,0,0,0,0,0]
]

@pytest.mark.parametrize("x",good_training_data)
def test_predict_happy(x):
    testobj.train(x)
    testobj.predict()

@pytest.mark.parametrize("x",bad_training_data)
def test_predict_fail(x):
    with pytest.raises((ValueError,AssertionError)):
        testobj.train(x)
        testobj.predict(x)
