"""
Mapping from values to their indices
====================================

IndexMap

"""
import numpy as np


def _to_key(v):
    """
    Convert a value to a hashable dict key: rows of a 2-dim array (and
    lists) become tuples, and NumPy scalars become Python scalars, so
    that keys hash consistently with plain Python queries such as
    `im[(0.0, 1.0)]`.

    """
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


class IndexMap:
    """
    Mapping from the values of a sequence to their indices: `im[v]`
    returns the index `i` such that `im.vals[i]` equals `v`, and raises
    an informative `KeyError` when `v` is not among the values.
    Construction requires the values to be unique (`ValueError`
    otherwise).

    Lookups use the `==`/`hash` semantics of the (converted) keys:
    query with values equal to those in the wrapped sequence. The
    sequence is held by reference, and mutating a stored value corrupts
    the map.

    Parameters
    ----------
    vals : array_like or range
        Sequence of unique values. May be a 1-dim array or any sequence
        of hashable values; a 2-dim array, in which case each *row* is
        a value and queries may be tuples, lists, or 1-dim arrays; or a
        `range`, in which case no dictionary is stored and lookups use
        `range.index` (O(1) arithmetic).

    Attributes
    ----------
    vals : array_like or range
        The wrapped sequence of values.

    Examples
    --------
    >>> im = IndexMap([(0.0, 0.1), (1.0, 0.1), (0.0, 1.0), (1.0, 1.0)])
    >>> im[(0.0, 1.0)]
    2
    >>> im = IndexMap(range(2000, 2020))
    >>> im[2005]
    5

    """
    def __init__(self, vals):
        self.vals = vals

        if isinstance(vals, range):
            self._dict = None
            return

        d = {}
        for i, v in enumerate(vals):
            key = _to_key(v)
            j = d.setdefault(key, i)
            if j != i:
                raise ValueError(
                    f'values must be unique: vals[{j}] and vals[{i}] '
                    f'are equal ({v!r})'
                )
        self._dict = d

    def __getitem__(self, v):
        if self._dict is None:  # self.vals is a range
            try:
                return self.vals.index(v)
            except (ValueError, TypeError):
                pass
        else:
            try:
                return self._dict[_to_key(v)]
            except (KeyError, TypeError):  # TypeError: unhashable query
                pass
        raise KeyError(
            f'value {v!r} is not among the values of this IndexMap'
        )

    def __contains__(self, v):
        if self._dict is None:
            try:
                return v in self.vals
            except TypeError:
                return False
        try:
            return _to_key(v) in self._dict
        except TypeError:
            return False

    def get(self, v, default=None):
        """
        Return the index of value `v`, or `default` if `v` is not among
        the values.

        """
        try:
            return self[v]
        except KeyError:
            return default

    def __len__(self):
        return len(self.vals)

    def __repr__(self):
        return f'IndexMap of length {len(self)}'
