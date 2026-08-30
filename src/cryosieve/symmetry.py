import re
from math import pi, sqrt
from typing import Tuple

import numpy as np


__all__ = ['get_point_group']


def _parse_grp_name(sym: str) -> Tuple[str, int]:
    pattern = r'^(C([1-9][0-9]*)|D([1-9][0-9]*)|T|O|I|I([1-4]))$'
    match = re.match(pattern, sym, re.IGNORECASE)
    if not match:
        raise ValueError(f'Invalid point group name {sym}.')

    grp = match.group(1)[0].upper()
    if grp == 'C':
        return grp, int(match.group(2))
    elif grp == 'D':
        return grp, int(match.group(3))
    elif grp in ['T', 'O']:
        return grp, 1
    else:
        return grp, 2 if match.group(4) is None else int(match.group(4))


def _quat_mult(q1, q2):
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    ], dtype=np.float64)


def _quat_same(q1, q2):
    return np.dot(q1, q2) ** 2 > 1 - 1e-6


def _generate_by_two(qa, qb):
    qe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elems = []

    def _recurse(q):
        if not any(_quat_same(q, e) for e in elems):
            elems.append(q)
            _recurse(_quat_mult(q, qa))
            _recurse(_quat_mult(q, qb))

    _recurse(qe)
    return np.array(elems, dtype=np.float64)


_T_ELEMS = _generate_by_two(
    np.array([0.5, 0.0, 0.0, sqrt(3) / 2], dtype=np.float64),
    np.array([0.0, 0.0, sqrt(6) / 3, sqrt(3) / 3], dtype=np.float64)
)

_O_ELEMS = _generate_by_two(
    np.array([sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2], dtype=np.float64),
    np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
)

_I1_ELEMS = _generate_by_two(
    np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.5, (sqrt(5) + 1) / 4, (sqrt(5) - 1) / 4, 0.0], dtype=np.float64)
)

_I2_ELEMS = _generate_by_two(
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.5, 0.0, (sqrt(5) - 1) / 4, (sqrt(5) + 1) / 4], dtype=np.float64)
)

_I3_ELEMS = _generate_by_two(
    np.array([0.0, -0.5257311143328698, 0.0, 0.8506508069838757], dtype=np.float64),
    np.array([0.5, -0.4253254059669702, 0.3090169956761489, 0.6881909591287187], dtype=np.float64)
)

_I_ELEMS = {1: _I1_ELEMS, 2: _I2_ELEMS, 3: _I3_ELEMS}


def get_point_group(sym: str) -> np.ndarray:
    '''Return point-group elements as unit quaternions.

    Supported groups are Cn, Dn, T, O, and I. The icosahedral conventions
    I1, I2, and I3 are supported, and I defaults to I2.

    Parameters
    ----------
    sym : str
        Point-group name, such as ``'C3'``, ``'D5'``, ``'T'``, ``'O'``,
        or ``'I'``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(M, 4)`` and dtype ``float64``. Each row is a unit
        quaternion ``(w, x, y, z)`` for one group element.

    Raises
    ------
    ValueError
        If the point-group name is invalid.
    NotImplementedError
        If the point group is recognized but not implemented, as for I4.
    '''
    grp, n = _parse_grp_name(sym)

    if grp == 'C':
        theta = np.linspace(0, pi, n, False)
        elems = np.zeros((n, 4), dtype=np.float64)
        elems[:, 0] = np.cos(theta)
        elems[:, 3] = np.sin(theta)
        return elems

    if grp == 'D':
        theta = np.linspace(0, pi, n, False)
        elems = np.zeros((2 * n, 4), dtype=np.float64)
        elems[:n, 0] = np.cos(theta)
        elems[:n, 3] = np.sin(theta)
        elems[n:, 1] = np.cos(theta)
        elems[n:, 2] = np.sin(theta)
        return elems

    if grp == 'T':
        return _T_ELEMS

    if grp == 'O':
        return _O_ELEMS

    if grp == 'I':
        if n not in _I_ELEMS:
            raise NotImplementedError(f'I{n} not implemented.')
        return _I_ELEMS[n]
