import pytest
import utils

testdata_non_increasing = [
    ([10,8,5,4,2,1],True),
    ([10,10,8,4,2,1], True),
    ([10,10],True),
    ([10,8],True),
    ([10,12],False),
    ([1,0,-4],True),
    ([10,4,-1,-4,-2],False)
]

@pytest.mark.parametrize("a,expected",testdata_non_increasing)
def test_is_non_increasing(a,expected):
    assert utils.is_non_increasing(a) == expected

    # Test exceptions
    with pytest.raises(ValueError):
        utils.is_non_increasing([1])
        utils.is_non_increasing([])

testdata_is_numeric = [
    ([10,1,2,3,-3],True),
    ([1,'Hi'], False),
    (['one','two',3,4],False),
    ([],False),
    (['string',-1,0.4,None],False),
    ([1,0.5,1],True)
]

@pytest.mark.parametrize("a,expected",testdata_is_numeric)
def test_is_numeric(a,expected):
    assert utils.is_numeric(a) == expected
