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



@pytest.fixture
def example_scope():
    return Scope("layer1")
