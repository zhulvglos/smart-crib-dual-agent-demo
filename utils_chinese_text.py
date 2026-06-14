"""
OpenCV 中文字体渲染工具模块。

OpenCV 的 cv2.putText() 只支持 ASCII 字符（Hershey 字体），
中文字符会显示为 "????"。本模块使用 PIL/Pillow 绕开此限制。

用法：
    from utils_chinese_text import cv2_puttext_cn

    # 与 cv2.putText() 相同的调用方式，但支持中文
    cv2_puttext_cn(frame, "婴儿床监护", (50, 30), size=24, color=(255,255,255))
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 字体配置
# ============================================================

# 按优先级尝试的字体列表（Path 对象）
_FONT_CANDIDATES = [
    # Windows 字体
    Path("C:/Windows/Fonts/msyh.ttc"),     # 微软雅黑（首选，清晰现代）
    Path("C:/Windows/Fonts/msyhbd.ttc"),   # 微软雅黑粗体
    Path("C:/Windows/Fonts/simhei.ttf"),   # 黑体
    Path("C:/Windows/Fonts/simkai.ttf"),   # 楷体
    Path("C:/Windows/Fonts/simsun.ttc"),   # 宋体
    Path("C:/Windows/Fonts/Deng.ttf"),     # DengXian
    # macOS 字体
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    # Linux 字体
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]

# 默认字体路径（首次调用时自动探测）
_DEFAULT_FONT_PATH: Optional[str] = None
# 字体缓存：{size: ImageFont} 避免重复加载
_FONT_CACHE: dict = {}


def _find_default_font() -> Optional[str]:
    """探测系统中第一个可用的中文字体。"""
    for path in _FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return None


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """获取指定大小的 PIL 字体对象（带缓存）。"""
    global _DEFAULT_FONT_PATH
    if _DEFAULT_FONT_PATH is None:
        _DEFAULT_FONT_PATH = _find_default_font()
    if _DEFAULT_FONT_PATH is None:
        raise FileNotFoundError(
            "未找到可用的中文字体文件！请安装微软雅黑(msyh.ttc)或黑体(simhei.ttf)"
        )

    cache_key = (_DEFAULT_FONT_PATH, size)
    if cache_key not in _FONT_CACHE:
        _FONT_CACHE[cache_key] = ImageFont.truetype(_DEFAULT_FONT_PATH, size)

    return _FONT_CACHE[cache_key]


def cv2_puttext_cn(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    size: int = 18,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_type: str = "solid",
    anchor: str = "lt",
):
    """
    在 OpenCV 图像上绘制中文文本（替代 cv2.putText）。

    Parameters
    ----------
    frame : np.ndarray
        OpenCV BGR 图像。
    text : str
        要绘制的文本（支持中英文混排）。
    position : Tuple[int, int]
        文字左上角坐标 (x, y)。
    size : int
        字体大小（像素）。默认 18。
    color : Tuple[int, int, int]
        RGB 颜色（BGR 格式）。默认白色 (255,255,255)。
    thickness : int
        描边粗细（仅当 line_type="stroke" 时生效）。
    line_type : str
        "solid" - 纯色文字（默认）
        "stroke" - 带描边文字（先画黑色描边再画彩色文字）
    anchor : str
        PIL 锚点。常用值：
        "lt" - 左上角（默认）
        "mm" - 中心
        "rb" - 右下角
        "mt" - 中上方
    """
    height, width = frame.shape[:2]
    font = _get_font(size)

    # numpy BGR → PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 颜色转换：BGR → RGB
    pil_color = (color[2], color[1], color[0])

    if line_type == "stroke":
        # 先画黑色描边
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                   (-1, 0), (1, 0), (0, -1), (0, 1)]
        for ox, oy in offsets:
            draw.text(
                (position[0] + ox, position[1] + oy),
                text, font=font, fill=(0, 0, 0), anchor=anchor
            )
        # 再画彩色文字
        draw.text(position, text, font=font, fill=pil_color, anchor=anchor)
    else:
        draw.text(position, text, font=font, fill=pil_color, anchor=anchor)

    # PIL RGB → numpy BGR（原地修改 frame）
    frame_rgb = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame[:, :, :] = frame_bgr
