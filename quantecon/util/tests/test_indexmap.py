"""
Tests for util/indexmap.py

"""
import numpy as np
import pytest
from numpy.testing import assert_

from quantecon.util import IndexMap


class TestIndexMap1Dim:
    def test_list(self):
        im = IndexMap([2000, 2005, 2010])
        assert_(im[2005] == 1)
        assert_(len(im) == 3)
        assert_(2010 in im)
        assert_(1999 not in im)
        assert_(im.get(1999) is None)
        assert_(im.get(2010, -1) == 2)

    def test_ndarray(self):
        im = IndexMap(np.linspace(0, 1, 11))
        assert_(im[0.5] == 5)
        assert_(im[np.float64(0.5)] == 5)

    def test_missing_raises_key_error(self):
        im = IndexMap([2000, 2005, 2010])
        with pytest.raises(KeyError):
            im[1999]

    def test_unhashable_query(self):
        im = IndexMap([1, 2, 3])
        with pytest.raises(KeyError):
            im[[1, 2]]  # Converted to tuple, then missing
        with pytest.raises(KeyError):
            im[{}]  # Unhashable and inconvertible -> KeyError
        assert_({} not in im)


class TestIndexMap2Dim:
    def setup_method(self):
        self.vals = np.array([[0., 0.1], [1., 0.1], [0., 1.], [1., 1.]])
        self.im = IndexMap(self.vals)

    def test_queries(self):
        assert_(self.im[(0., 1.)] == 2)          # Tuple
        assert_(self.im[[0., 1.]] == 2)          # List
        assert_(self.im[np.array([0., 1.])] == 2)  # 1-dim array
        assert_(self.im[self.vals[3]] == 3)      # Row of the array

    def test_missing(self):
        with pytest.raises(KeyError):
            self.im[(2., 2.)]
        assert_((2., 2.) not in self.im)

    def test_tuples(self):
        vals = [(0.0, 0.1), (1.0, 0.1), (0.0, 1.0), (1.0, 1.0)]
        im = IndexMap(vals)
        assert_(im[(0.0, 1.0)] == 2)


class TestIndexMapRange:
    def test_lookup(self):
        im = IndexMap(range(2000, 2020))
        assert_(im._dict is None)  # Dict-free specialization
        assert_(im[2005] == 5)
        assert_(2019 in im)
        assert_(2020 not in im)
        assert_(len(im) == 20)
        with pytest.raises(KeyError):
            im[2020]

    def test_nonunit_step(self):
        im = IndexMap(range(0, 10, 3))
        assert_(im[6] == 2)
        with pytest.raises(KeyError):
            im[5]

    def test_non_number_query(self):
        im = IndexMap(range(3))
        with pytest.raises(KeyError):
            im['a']
        assert_('a' not in im)


class TestIndexMapUniqueness:
    def test_duplicates_error_indices(self):
        with pytest.raises(ValueError, match=r'vals\[0\] and vals\[2\]'):
            IndexMap([1, 2, 1])

    def test_duplicates_2dim(self):
        with pytest.raises(ValueError, match='unique'):
            IndexMap(np.array([[0., 1.], [0., 1.]]))
