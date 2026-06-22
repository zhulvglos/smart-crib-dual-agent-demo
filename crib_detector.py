"""
自动检测婴儿床区域的模块。
使用 YOLO 实例分割模型自动检测床的轮廓，
并自动计算安全区、警告区和危险区。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class CribGeometry:
    """床的几何信息"""
    # 床的轮廓点（像素坐标）
    crib_contour: np.ndarray
    # 安全区（向内收缩）
    safe_contour: np.ndarray
    # 警告区（床轮廓本身
    warning_contour: np.ndarray
    # 边界框
    bbox: Tuple[int, int, int, int]


@dataclass
class DetectionResult:
    """检测结果"""
    success: bool
    geometry: Optional[CribGeometry] = None
    method: str = ""
    message: str = ""
    confidence: float = 0.0


class CribDetector:
    """婴儿床检测器"""
    
    def __init__(self, seg_model_path: str = "yolo26n-seg.pt"):
        self.seg_model_path = seg_model_path
        self._model = None
        self._fallback_model = None
    
    @property
    def model(self):
        self._load_models()
        return self._model
    
    @property
    def fallback_model(self):
        self._load_models()
        return self._fallback_model
    
    def _load_models(self):
        """加载模型（延迟加载）"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics 模块未安装，请运行 `pip install ultralytics`")
        
        if self._model is None:
            try:
                self._model = YOLO(self.seg_model_path)
            except Exception as e:
                print(f"[警告] 加载分割模型 {self.seg_model_path} 失败: {e}")
                self._model = None
        
        if self._fallback_model is None:
            try:
                self._fallback_model = YOLO("yolo11n.pt")
            except Exception as e:
                print(f"[警告] 加载备用检测模型失败: {e}")
                self._fallback_model = None
    
    def detect_crib(
        self,
        frame: np.ndarray,
        safe_ratio: float = 0.15,
        conf_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        检测床区域
        
        Args:
            frame: 输入帧图像
            safe_ratio: 安全区向内收缩比例
            conf_threshold: 检测置信度阈值
        """
        height, width = frame.shape[:2]
        
        # 先检测婴儿位置（用于动态确定危险边界）
        baby_center = self._detect_baby_center(frame)
        
        # 尝试1: 用分割模型检测床
        if self.model is not None:
            try:
                result = self._try_detect_with_seg(frame, width, height, safe_ratio, conf_threshold, baby_center)
                if result.success:
                    return result
            except Exception as e:
                print(f"[警告] 分割检测失败: {e}")
        
        # 尝试2: 降级方案，用检测模型检测人
        if self.fallback_model is not None:
            try:
                result = self._try_detect_with_person(frame, width, height, safe_ratio, baby_center)
                if result.success:
                    return result
            except Exception as e:
                print(f"[警告] 人员检测失败: {e}")
        
        # 最终降级: 使用硬编码的矩形区域
        result = self._fallback_to_rect(width, height, safe_ratio, baby_center)
        return result
    
    def _detect_baby_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """检测婴儿位置（用于动态确定危险边界）"""
        if self.fallback_model is None:
            return None
        
        try:
            results = self.fallback_model.predict(frame, classes=[0], conf=0.3, verbose=False)
            if results and results[0].boxes:
                boxes = results[0].boxes
                # 取最大的人（假设是婴儿）
                person_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1, y1, x2, y2 = person_box.xyxy[0].tolist()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return (center_x, center_y)
        except Exception as e:
            print(f"[警告] 婴儿检测失败: {e}")
        
        return None
    
    def _try_detect_with_seg(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        safe_ratio: float,
        conf_threshold: float,
        baby_center: Optional[Tuple[int, int]] = None,
    ) -> DetectionResult:
        """尝试用分割模型检测"""
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            verbose=False,
        )
        
        if not results or len(results) == 0:
            return DetectionResult(success=False, message="无检测结果")
        
        result = results[0]
        if not result.masks:
            return DetectionResult(success=False, message="无 mask 结果")
        
        # 寻找床的 mask
        best_mask = None
        best_conf = 0.0
        
        for i, box in enumerate(result.boxes):
            if result.names[int(box.cls[0])] == "bed":
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_mask = result.masks.data[i].cpu().numpy()
        
        if best_mask is None:
            # 如果没找到床，用最大的物体
            largest_result = self._find_largest_mask(result.masks.data.cpu().numpy())
            if largest_result is not None:
                best_mask, best_conf = largest_result
        
        if best_mask is None:
            return DetectionResult(success=False, message="未找到床或合适物体")
        
        # 处理 mask
        geometry = self._process_mask(best_mask, width, height, safe_ratio, baby_center)
        return DetectionResult(
            success=True,
            geometry=geometry,
            method="segmentation",
            confidence=best_conf,
        )
    
    def _find_largest_mask(
        self,
        masks: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """找最大的 mask"""
        if masks.ndim == 3:
            max_idx = np.argmax([m.sum() for m in masks])
            return masks[max_idx], 0.5
        return None
    
    def _process_mask(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
        safe_ratio: float,
        baby_center: Optional[Tuple[int, int]] = None,
    ) -> CribGeometry:
        """处理 mask 得到轮廓"""
        # 调整 mask 到原始尺寸
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height))
        
        # 二值化
        binary = ((mask > 0.5).astype(np.uint8)) * 255
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            raise RuntimeError("未找到轮廓")
        
        # 取最大轮廓
        crib_contour = max(contours, key=cv2.contourArea)
        
        # 简化轮廓为四边形
        simplified = self._simplify_to_quadrilateral(crib_contour)
        
        return self._create_geometry_from_contour(simplified, width, height, safe_ratio, baby_center)
    
    def _simplify_to_quadrilateral(self, contour: np.ndarray) -> np.ndarray:
        """简化轮廓为四边形"""
        # 用多边形逼近
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果顶点数不是4，尝试调整epsilon
        for _ in range(10):
            if len(approx) == 4:
                break
            epsilon *= 1.1
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            # 还是不行就用 bounding box 的四个角
            x, y, w, h = cv2.boundingRect(contour)
            approx = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]],
            ], dtype=np.int32)
        
        # 确保顺序正确
        approx = self._order_points_clockwise(approx)
        return approx
    
    def _order_points_clockwise(self, points: np.ndarray) -> np.ndarray:
        """将点按顺时针排序"""
        pts = points.reshape(-1, 2)
        # 计算中心
        center = np.mean(pts, axis=0)
        # 计算角度
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        # 排序
        sorted_idx = np.argsort(angles)
        return pts[sorted_idx].reshape(-1, 1, 2)
    
    def _create_geometry_from_contour(
        self,
        contour: np.ndarray,
        width: int,
        height: int,
        safe_ratio: float,
        baby_center: Optional[Tuple[int, int]] = None,
    ) -> CribGeometry:
        """从轮廓创建几何信息"""
        # 计算安全区（向内收缩）
        safe_contour = self._shrink_contour(contour, safe_ratio)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        return CribGeometry(
            crib_contour=contour,
            safe_contour=safe_contour,
            warning_contour=contour,
            bbox=(x, y, w, h),
        )
    
    def _shrink_contour(self, contour: np.ndarray, ratio: float) -> np.ndarray:
        """向内收缩轮廓"""
        # 简单方法：用 bounding box 收缩
        x, y, w, h = cv2.boundingRect(contour)
        shrink_x = int(w * ratio)
        shrink_y = int(h * ratio)
        new_x = x + shrink_x
        new_y = y + shrink_y
        new_w = w - 2 * shrink_x
        new_h = h - 2 * shrink_y
        new_w = max(new_w, 20)
        new_h = max(new_h, 20)
        
        return np.array([
            [[new_x, new_y]],
            [[new_x + new_w, new_y]],
            [[new_x + new_w, new_y + new_h]],
            [[new_x, new_y + new_h]],
        ], dtype=np.int32).reshape(-1, 1, 2)
    
    def _try_detect_with_person(
        self, 
        frame: np.ndarray, 
        width: int, 
        height: int, 
        safe_ratio: float,
        baby_center: Optional[Tuple[int, int]] = None
    ) -> DetectionResult:
        """用检测人的位置估算"""
        results = self.fallback_model.predict(frame, classes=[0], conf=0.3, verbose=False)
        if not results or not results[0].boxes:
            return DetectionResult(success=False, message="未检测到人")
        
        boxes = results[0].boxes
        # 取最大的人
        person_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = person_box.xyxy[0].tolist()
        
        # 以人中心为基准，向外扩展估算床区域
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        crib_w = int(width * 0.6)
        crib_h = int(height * 0.7)
        crib_x = int(max(0, center_x - crib_w / 2))
        crib_y = int(max(0, center_y - crib_h / 2))
        
        # 创建矩形轮廓
        contour = np.array([
            [[crib_x, crib_y]],
            [[crib_x + crib_w, crib_y]],
            [[crib_x + crib_w, crib_y + crib_h]],
            [[crib_x, crib_y + crib_h]],
        ], dtype=np.int32)
        
        geometry = self._create_geometry_from_contour(contour, width, height, safe_ratio, baby_center)
        
        return DetectionResult(
            success=True,
            geometry=geometry,
            method="person_estimation",
            confidence=float(person_box.conf[0]),
        )
    
    def _fallback_to_rect(
        self, 
        width: int, 
        height: int, 
        safe_ratio: float,
        baby_center: Optional[Tuple[int, int]] = None
    ) -> DetectionResult:
        """最终降级：硬编码的矩形"""
        x = int(width * 0.15)
        y = int(height * 0.1)
        w = int(width * 0.7)
        h = int(height * 0.8)
        
        contour = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]],
        ], dtype=np.int32).reshape(-1, 1, 2)
        
        geometry = self._create_geometry_from_contour(contour, width, height, safe_ratio, baby_center)
        
        return DetectionResult(
            success=True,
            geometry=geometry,
            method="hardcoded",
            confidence=0.0,
        )


def save_crib_config(
    config_path: Path,
    video_path: str,
    geometry: CribGeometry,
    width: int,
    height: int,
):
    """保存床配置到文件"""
    # 转换为比例坐标
    corners_ratio = []
    for point in geometry.crib_contour:
        x, y = point[0]
        corners_ratio.append({"x": x / width, "y": y / height})
    
    # 读取现有配置
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    
    # 更新配置（路径统一用正斜杠）
    config["auto_crib"] = {
        "enabled": True,
        "video_path": video_path.replace("\\", "/"),
        "corners_ratio": corners_ratio,
        "corner_order": "top_left,top_right,bottom_right,bottom_left",
    }
    
    # 计算矩形区域（用于兼容
    safe_x1 = min(p[0][0] for p in geometry.safe_contour) / width
    safe_y1 = min(p[0][1] for p in geometry.safe_contour) / height
    safe_x2 = max(p[0][0] for p in geometry.safe_contour) / width
    safe_y2 = max(p[0][1] for p in geometry.safe_contour) / height
    
    config["safe_zone"] = {
        "x1_ratio": safe_x1,
        "y1_ratio": safe_y1,
        "x2_ratio": safe_x2,
        "y2_ratio": safe_y2,
    }
    
    warning_x1 = min(p[0][0] for p in geometry.warning_contour) / width
    warning_y1 = min(p[0][1] for p in geometry.warning_contour) / height
    warning_x2 = max(p[0][0] for p in geometry.warning_contour) / width
    warning_y2 = max(p[0][1] for p in geometry.warning_contour) / height
    
    config["warning_zone"] = {
        "x1_ratio": warning_x1,
        "y1_ratio": warning_y1,
        "x2_ratio": warning_x2,
        "y2_ratio": warning_y2,
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_crib_config(
    config_path: Path,
    video_path: str,
    width: int,
    height: int,
) -> Optional[CribGeometry]:
    """从配置加载床配置"""
    if not config_path.exists():
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    auto_crib = config.get("auto_crib", {})
    if not auto_crib.get("enabled"):
        return None

    # 规范化路径：统一用正斜杠，消除 Windows 反斜杠 vs 正斜杠差异
    saved_path = os.path.normpath(auto_crib.get("video_path", "")).replace("\\", "/")
    input_path = os.path.normpath(video_path).replace("\\", "/")
    # Demo media may be copied between data/ and web_demo/data/. Reuse the
    # calibrated geometry when the normalized path or filename still matches.
    if saved_path != input_path and Path(saved_path).name != Path(input_path).name:
        return None
    
    corners_ratio = auto_crib.get("corners_ratio", [])
    if len(corners_ratio) != 4:
        return None
    
    # 转换为像素坐标
    contour = []
    for pt in corners_ratio:
        x = int(pt["x"] * width)
        y = int(pt["y"] * height)
        contour.append([[x, y]])
    contour = np.array(contour, dtype=np.int32)
    
    # 重建几何
    safe_ratio = 0.15
    detector = CribDetector()
    return detector._create_geometry_from_contour(contour, width, height, safe_ratio)
