"""图像暗角（vignette）核心实现。

设计要点
--------
- 因子模型：``factor = 1 - strength * r2``，其中 ``r2`` 为归一化平方半径
  （中心为 0，四角恰好为 1）。中心像素因子恒为 1（中心不变）；因子随
  strength 单调不增、随半径单调不增。
- LUT 快速路径：``build_lut`` 预计算 ``r2 -> factor`` 的一维表（不可变
  tuple，可跨线程、跨调用安全复用）。因子关于 r2 是线性的，因此对 LUT
  做线性插值与逐像素直接计算的差异仅为浮点舍入量级（远小于 1e-6）。
- 内存：``apply`` 按行流式处理，仅持有 O(宽) 的列因子表与 O(宽) 的单行
  缓冲，不展开与图像同尺寸的浮点缓冲；4K 图的额外峰值内存远低于 8 MB。
- 并发安全：不保存任何可变共享状态，多线程并发调用 ``apply`` 安全。
"""

from __future__ import annotations

import math

__all__ = ["Vignette"]

_DEFAULT_LUT_SIZE = 4096


class Vignette:
    """对 ``list[list[float]]`` 图像应用暗角。"""

    def __init__(self, lut_size=_DEFAULT_LUT_SIZE):
        if not isinstance(lut_size, int) or isinstance(lut_size, bool) or lut_size < 2:
            raise ValueError("lut_size 必须为 >= 2 的整数")
        self._lut_size = lut_size

    # ------------------------------------------------------------------
    # LUT 快速路径
    # ------------------------------------------------------------------
    def build_lut(self, strength):
        """预计算 ``r2 -> 因子`` 的 LUT。

        返回不可变 tuple，可在多次 ``apply`` 调用及多线程间共享。
        """
        s = self._check_strength(strength)
        n = self._lut_size
        return tuple(1.0 - s * (i / n) for i in range(n + 1))

    @staticmethod
    def _lut_at(lut, t):
        """对 LUT 做线性插值（t ∈ [0,1]）。因子关于 t 线性，插值无损。"""
        n = len(lut) - 1
        x = t * n
        i = int(x)
        if i >= n:
            return lut[n]
        frac = x - i
        return lut[i] + (lut[i + 1] - lut[i]) * frac

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def apply(self, image, strength, lut=None):
        """应用暗角，返回新的 list[list[float]]，不修改输入。

        - ``strength`` 必须在 [0, 1] 内且有限，否则抛出 ValueError；
        - 图像必须非空、各行等宽，像素必须为有限数值，否则抛出 ValueError；
        - ``lut`` 可选：传入 ``build_lut(strength)`` 的预计算结果以走
          快速路径（与逐像素计算结果差 <= 1e-6）。
        """
        s = self._check_strength(strength)
        rows = self._check_image(image)
        h = len(rows)
        w = len(rows[0])
        if lut is None:
            lut = self.build_lut(s)
        else:
            try:
                if len(lut) < 2:
                    raise ValueError("lut 至少需要 2 个表项")
            except TypeError:
                raise ValueError("lut 必须为数值序列") from None

        wd = (w - 1) or 1
        hd = (h - 1) or 1
        # 因子关于 r2 线性，故可分解：
        #   factor(x, y) = L((dx2+dy2)/2) = L(dx2/2) + (L(dy2/2) - 1)
        # 列方向查表一次（O(w)），行方向每行一个标量，避免逐像素求半径，
        # 也无需任何与图像同尺寸的浮点缓冲。
        col = [
            self._lut_at(lut, min((((2.0 * x - (w - 1)) / wd) ** 2) * 0.5, 1.0))
            for x in range(w)
        ]
        out = []
        for y in range(h):
            row = rows[y]
            dy2 = ((2.0 * y - (h - 1)) / hd) ** 2
            rowterm = self._lut_at(lut, min(dy2 * 0.5, 1.0)) - 1.0
            try:
                finite = all(map(math.isfinite, row))
            except TypeError:
                raise ValueError("像素必须为有限数值") from None
            if not finite:
                raise ValueError("像素必须为有限数值（不允许 NaN/inf）")
            fac = [c + rowterm for c in col]
            out.append([p * f for p, f in zip(row, fac)])
        return out

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @staticmethod
    def _check_strength(strength):
        try:
            s = float(strength)
        except (TypeError, ValueError):
            raise ValueError(
                "strength 必须为 [0, 1] 内的实数，收到 %r" % (strength,)
            ) from None
        if not math.isfinite(s) or s < 0.0 or s > 1.0:
            raise ValueError("strength 越界（须在 [0, 1] 内）：%r" % (strength,))
        return s

    @staticmethod
    def _check_image(image):
        try:
            rows = list(image)
        except TypeError:
            raise ValueError("图像必须为非空的行序列") from None
        if not rows:
            raise ValueError("图像不能为空")
        try:
            width = len(rows[0])
        except TypeError:
            raise ValueError("图像行必须为序列") from None
        if width == 0:
            raise ValueError("图像行不能为空")
        for row in rows:
            try:
                n = len(row)
            except TypeError:
                raise ValueError("图像行必须为序列") from None
            if n != width:
                raise ValueError("图像各行宽度不一致")
        return rows
