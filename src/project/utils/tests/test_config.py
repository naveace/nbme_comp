import pytest
from pytest import raises
from project.utils.config import Config
def test_creation():
    d = {'a': 1, 'b': 2}
    c = Config(d)
    assert c['a'] == 1
    assert c['b'] == 2
    assert c.get('a') == 1
    assert c.get('b') == 2
    d['c'] = 3
    assert 'c' not in c
    assert c.get('c') is None
    pytest.raises(KeyError, lambda: c['c'])
    assert list(c.items()) == [('a', 1), ('b', 2)]

