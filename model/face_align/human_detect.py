import cv2
import torch
from typing import List, Dict, Optional
import numpy as np
import os
from ultralytics import YOLO

class YOLOv8HumanDetector:
    """
    YOLOv8n 人体检测器，输入图片返回按框面积降序排列的人体标注框信息。
    功能：
        1. 加载预训练的 YOLOv8n 模型
        2. 执行目标检测（默认只检测 'person' 类别）
        3. 返回按检测框面积降序排列的结果
    """

    def __init__(self, model_path: str = 'yolov8n.pt', device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        self.model = self._load_model(model_path)
        self.class_names = self.model.names  # YOLOv8 直接通过 model.names 获取类别名

    def _load_model(self, model_path: str):
        model = YOLO(model_path)  # 加载 YOLOv8 模型
        model.to(self.device).eval()
        return model

    def _calculate_area(self, bbox: List[int]) -> int:
        """计算检测框面积 (宽度 * 高度)"""
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        检测图片中的人体并返回按框面积降序排列的标注框信息
        :param image: 输入图片 (BGR格式)
        :param conf_threshold: 置信度阈值
        :return: 检测结果列表，按面积从大到小排序
        """

        results = self.model(image, verbose=False)  # YOLOv8 推理
        # print("原始检测结果:", results[0].boxes.data)  # 查看所有检测框数据
        # print("模型类别名称及ID:", self.class_names)

        detections = []
        if results[0].boxes is not None:  # 检查是否有检测结果
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框坐标
            confs = results[0].boxes.conf.cpu().numpy()  # 获取置信度
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 获取类别ID

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id == 0 and conf >= conf_threshold:  # 只保留 'person' 类别 (cls_id=0)
                    bbox = [int(coord) for coord in box]
                    detections.append({
                        'bbox': bbox,
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.class_names[cls_id],
                        'area': self._calculate_area(bbox)
                    })

        # 按面积降序排序
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections

    def visualize(self, image: np.ndarray, detections: List[Dict], thickness: int = 2) -> np.ndarray:
        """绘制检测框（框颜色按面积大小渐变：面积越大框越绿）"""
        output_image = image.copy()
        max_area = max(det['area'] for det in detections) if detections else 1

        for detection in detections:
            x_min, y_min, x_max, y_max = detection['bbox']
            green_value = int(255 * (detection['area'] / max_area))
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max),
                          (0, green_value, 0), thickness)
            label = f"{detection['class_name']} {detection['confidence']:.2f} (area={detection['area']})"
            cv2.putText(output_image, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, green_value, 0), 1)
        return output_image


# 使用示例
if __name__ == "__main__":
    detector = YOLOv8HumanDetector()  # 使用默认的 yolov8n.pt
    image = cv2.imread('../video/000002.jpg')
    assert image is not None, "图片读取失败！"

    detections = detector.detect(image)
    print(f"检测到 {len(detections)} 个人体（按面积降序排列）：")
    for i, det in enumerate(detections, 1):
        print(f"人体 {i}: 位置={det['bbox']}, 面积={det['area']}, 置信度={det['confidence']:.2f}")

    result_image = detector.visualize(image, detections)
    cv2.imwrite('result_sorted.jpg', result_image)
    cv2.imshow('Sorted Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
