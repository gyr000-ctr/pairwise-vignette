import math

import pytest

import vignette

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[1.0] * 5 for _ in range(5)]


def _v():
    return vignette.Vignette()


def _check_zero_strength():
    return _v().apply(IMG, 0.0) == IMG


def _check_center():
    out = _v().apply(IMG, 1.0)
    return out[2][2] == 1.0


def _check_corner_darker():
    out = _v().apply(IMG, 1.0)
    return out[0][0] < out[2][2]


def _check_monotonic():
    a = _v().apply(IMG, 0.5)[0][0]
    b = _v().apply(IMG, 1.0)[0][0]
    return b <= a


def _check_bad_strength():
    return _raises(lambda: _v().apply(IMG, 2.0))


def _check_bad_image():
    return _raises(lambda: _v().apply([], 1.0))


def _check_shape():
    out = _v().apply(IMG, 1.0)
    return len(out) == 5 and len(out[0]) == 5


def _check_range():
    out = _v().apply(IMG, 1.0)
    return all(0.0 <= v <= 1.0 for row in out for v in row)


def _check_det():
    return _v().apply(IMG, 1.0) == _v().apply(IMG, 1.0)


def _check_type():
    return isinstance(_v().apply(IMG, 0.5), list)


def _check_no_mutation():
    src = [row[:] for row in IMG]
    _v().apply(IMG, 1.0)
    return IMG == src


def _check_negative():
    return _raises(lambda: _v().apply(IMG, -1.0))


def _check_center_strength():
    out = _v().apply(IMG, 0.25)
    return out[2][2] == 1.0


def _check_edges():
    out = _v().apply(IMG, 1.0)
    return out[0][2] <= 1.0 and out[2][0] <= 1.0


def _check_small_image():
    return _v().apply([[1.0]], 1.0) == [[1.0]]

CASES = [("zero_strength", "_check_zero_strength"), ("center", "_check_center"), ("corner_darker", "_check_corner_darker"), ("monotonic", "_check_monotonic"), ("bad_strength", "_check_bad_strength"), ("bad_image", "_check_bad_image"), ("shape", "_check_shape"), ("range", "_check_range"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("negative", "_check_negative"), ("center_strength", "_check_center_strength"), ("edges", "_check_edges"), ("small_image", "_check_small_image")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
