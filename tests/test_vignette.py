"""暗角实现的边界与约束自测（不改动既有 test_core.py）。"""

import math
import threading
import tracemalloc

import pytest

import vignette


def _img(w, h, value=0.8):
    return [[value] * w for _ in range(h)]


def test_vignette_lut_matches_per_pixel():
    """LUT 快速路径与逐像素直接计算差 <= 1e-6（非方形图）。"""
    v = vignette.Vignette()
    lut = v.build_lut(0.7)
    img = [[0.5 + 0.001 * (x + 3 * y) for x in range(61)] for y in range(37)]
    out = v.apply(img, 0.7, lut=lut)
    h, w = len(img), len(img[0])
    for y in range(h):
        for x in range(w):
            dx = (2.0 * x - (w - 1)) / (w - 1)
            dy = (2.0 * y - (h - 1)) / (h - 1)
            r2 = min((dx * dx + dy * dy) / 2.0, 1.0)
            expect = img[y][x] * (1.0 - 0.7 * r2)
            assert abs(out[y][x] - expect) <= 1e-6


def test_vignette_lut_reuse_consistent():
    """预计算 LUT 复用与自动建表结果一致。"""
    v = vignette.Vignette()
    img = _img(33, 17)
    lut = v.build_lut(0.9)
    assert v.apply(img, 0.9, lut=lut) == v.apply(img, 0.9)


def test_vignette_nonfinite_pixels_rejected():
    v = vignette.Vignette()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            v.apply([[1.0, 1.0], [1.0, bad]], 0.5)


def test_vignette_strength_bounds():
    v = vignette.Vignette()
    img = [[1.0]]
    v.apply(img, 0.0)
    v.apply(img, 1.0)
    for bad in (-1e-9, 1.0000001, float("nan"), float("inf"), None):
        with pytest.raises(ValueError):
            v.apply(img, bad)


def test_vignette_non_square_center_and_monotonic_radius():
    v = vignette.Vignette()
    out = v.apply(_img(7, 3), 1.0)
    assert len(out) == 3 and len(out[0]) == 7
    assert out[1][3] == 0.8  # 中心不变
    assert out[0][0] <= out[0][3] <= out[1][3]  # 半径越大越暗


def test_vignette_concurrent_apply():
    """多线程共享同一实例与同一 LUT，结果确定且一致。"""
    v = vignette.Vignette()
    lut = v.build_lut(0.6)
    img = _img(64, 64)
    results = [None] * 8

    def work(i):
        results[i] = v.apply(img, 0.6, lut=lut)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == results[0] for r in results)


def test_vignette_streaming_memory():
    """行流式处理：除必要返回值外不展开同尺寸浮点缓冲。

    512x512 输出的必要浮点+列表开销约 9 MB；若实现额外展开同尺寸
    浮点缓冲，峰值将明显超过该量级的两倍。这里断言峰值有界。
    """
    v = vignette.Vignette()
    img = _img(512, 512)
    tracemalloc.start()
    try:
        out = v.apply(img, 0.5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out is not None
    assert peak < 24 * 1024 * 1024


def test_vignette_large_image_timing():
    """大图（约 4K 的 1/4 规模）应在可接受时间内完成。"""
    import time

    v = vignette.Vignette()
    img = _img(1920, 1080)
    t0 = time.perf_counter()
    out = v.apply(img, 0.5)
    dt = time.perf_counter() - t0
    assert out[0][0] < out[540][960]
    assert dt < 10.0
