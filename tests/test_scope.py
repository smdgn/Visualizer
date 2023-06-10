import unittest
import pytest

from src.scope import Scope


# things to test:
# scope arithmetic +,-,*
# scope init
# scope.get_model_layers based on scope type
# scope get_item op
# scope batch bundling

# class MyTestCase(unittest.TestCase):
# def test_something(self):
# self.assertEqual(True, False)  # add assertion here


# if __name__ == '__main__':
# unittest.main()
# ===== Fixtures ==================#
@pytest.fixture
def example_scope1():
    return Scope("layer1")


@pytest.fixture
def example_scope2():
    return Scope("layer2")

# ==================================== #


@pytest.mark.parametrize("scope_init_name, expected_scope_name",
                         [
                             ("", [""]),
                             ("layer1", ["layer1"]),
                             (["layer1", "layer2"], ["layer1", "layer2"])
                         ])
def test_scope_valid_init(scope_init_name, expected_scope_name):
    assert Scope(scope_init_name).layer == expected_scope_name


@pytest.mark.parametrize("invalid_scope_name_structure",
                         [
                             [["layer1", "layer2"], ["layer3"]],
                         ])
def test_scope_invalid_init(invalid_scope_name_structure):
    with pytest.raises(ValueError) :
        Scope(invalid_scope_name_structure)



@pytest.mark.arithmetic
def test_scope_arithmetic_plus(example_scope1, example_scope2):
    assert example_scope1 + example_scope2 == \
       {
            1: {"item": example_scope1,
                "weight": 1.0},
            2: {"item": example_scope2,
                "weight": 1.0}
       }


@pytest.mark.arithmetic
def test_scope_arithmetic_plus_rev(example_scope1, example_scope2):
    assert example_scope1 + example_scope2 == \
           {
               1: {"item": example_scope2,
                   "weight": 1.0},
               2: {"item": example_scope1,
                   "weight": 1.0}
           }


@pytest.mark.arithmetic
def test_scope_arithmetic_sub(example_scope1, example_scope2):
    assert example_scope1 - example_scope2 == \
           {
               1: {"item": example_scope1,
                   "weight": 1.0},
               2: {"item": example_scope2,
                   "weight": -1.0}
           }


@pytest.mark.arithmetic
@pytest.mark.parametrize("example_numbers_valid",
                         [1.0,
                          2.0,
                          1,
                          -1,
                          -2.0,
                          +1,
                          +2.0
                          ])
def test_scope_arithmetic_mul_valid(example_scope1, example_numbers_valid):
    example_scope1*example_numbers_valid
    assert example_scope1._loss_weight == float(example_numbers_valid)


@pytest.mark.arithmetic
@pytest.mark.parametrize("example_numbers_invalid",
                         ["str",
                          example_scope2
                          ])
def test_scope_arithmetic_mul_invalid(example_scope1, example_numbers_invalid):
    with pytest.raises(ValueError):
        example_scope1*example_numbers_invalid

