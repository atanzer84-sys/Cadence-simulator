from poc.proof_of_concept import add


def test_add():
    assert add(2, 3) == 5

def test_add_not_equal():
    assert add(2, 2) != 1