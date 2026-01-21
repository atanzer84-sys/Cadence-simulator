from poc.proof_of_concept import add, multiply


def test_add():
    assert add(2, 3) == 5

def test_add_not_equal():
    assert add(2, 2) != 1


def test_multiply():
    assert multiply(3, 4) == 12


def test_multiply_with_zero():
    assert multiply(0, 5) == 0