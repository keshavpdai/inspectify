import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLO11mDetector:
    def __init__(
        self, model_path: str = "models/yolo11m_trained.pt", device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.class_names = {
            0: "dent",
            1: "scratch",
            2: "crack",
            3: "broken_lamp",
            4: "shattered_glass",
            5: "flat_tire",
        }

        logger.info(f"YOLO11m loaded on {self.device}")

    def predict(
        self,
        image_array: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 1280,
        augment: bool = False,
        agnostic_nms: bool = False,
        class_thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        results = self.model(
            image_array,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )

        if not results:
            return [], image_array

        result = results[0]
        detections = []

        for box in result.boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": self.class_names.get(int(box.cls[0]), "unknown"),
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x_min": float(box.xyxy[0][0]),
                    "y_min": float(box.xyxy[0][1]),
                    "x_max": float(box.xyxy[0][2]),
                    "y_max": float(box.xyxy[0][3]),
                    "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                    "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                },
                "pixel_area": int(
                    (box.xyxy[0][2] - box.xyxy[0][0])
                    * (box.xyxy[0][3] - box.xyxy[0][1])
                ),
            }
            th = None
            if class_thresholds is not None:
                th = class_thresholds.get(detection["class_name"])  
            if th is not None and detection["confidence"] < th:
                pass
            else:
                detections.append(detection)

        annotated_image = result.plot()
        return detections, annotated_image

    def predict_with_offset(
        self, 
        image_array: np.ndarray, 
        bbox_offset: Tuple[int, int] = (0, 0),
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 1280,
        augment: bool = False,
        agnostic_nms: bool = False,
        class_thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Run damage detection on cropped image with coordinate adjustment
        
        Args:
            image_array: Cropped vehicle region
            bbox_offset: (x_offset, y_offset) to adjust back to full image coords
            conf: Confidence threshold for damage detection
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            augment: Enable test-time augmentation
            agnostic_nms: Class-agnostic NMS
            class_thresholds: Per-class confidence thresholds
            
        Returns:
            (detections_list, annotated_image)
            Detections have coordinates adjusted to original image space
        """
        results = self.model(
            image_array,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )
        
        if not results:
            return [], image_array
        
        result = results[0]
        detections = []
        
        x_offset, y_offset = bbox_offset
        
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.class_names.get(int(box.cls[0]), 'unknown'),
                'confidence': float(box.conf[0]),
                'bbox': {
                    # Adjust coordinates back to full image space
                    'x_min': float(box.xyxy[0][0]) + x_offset,
                    'y_min': float(box.xyxy[0][1]) + y_offset,
                    'x_max': float(box.xyxy[0][2]) + x_offset,
                    'y_max': float(box.xyxy[0][3]) + y_offset,
                    'width': float(box.xyxy[0][2] - box.xyxy[0][0]),
                    'height': float(box.xyxy[0][3] - box.xyxy[0][1])
                },
                'pixel_area': int((box.xyxy[0][2] - box.xyxy[0][0]) * 
                                 (box.xyxy[0][3] - box.xyxy[0][1]))
            }
            
            # Apply class-specific thresholds if provided
            th = None
            if class_thresholds is not None:
                th = class_thresholds.get(detection['class_name'])
            if th is not None and detection['confidence'] < th:
                pass
            else:
                detections.append(detection)
        
        annotated_image = result.plot()
        return detections, annotated_image

    def create_annotated_image(
        self, 
        image: np.ndarray, 
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Create annotated image from filtered detections
        
        This ensures the annotated image matches the filtered detections
        in the API response, preventing mismatches between image and JSON.
        
        Args:
            image: Original image
            detections: Filtered detection list
            
        Returns:
            Annotated image with bounding boxes
        """
        import cv2
        
        annotated = image.copy()
        
        # Color map for each class (BGR format)
        # Using colors with good contrast for white text
        colors = {
            "dent": (0, 0, 255),           # Red
            "scratch": (255, 0, 0),         # Blue
            "crack": (0, 140, 255),         # Dark Orange (good contrast with white)
            "broken_lamp": (255, 0, 255),   # Magenta
            "shattered_glass": (0, 200, 0), # Dark Green (better than bright green)
            "flat_tire": (0, 100, 255),     # Orange-Red
        }
        
        # Text colors for each class (for better readability)
        text_colors = {
            "dent": (255, 255, 255),           # White on Red
            "scratch": (255, 255, 255),        # White on Blue
            "crack": (255, 255, 255),          # White on Dark Orange
            "broken_lamp": (255, 255, 255),    # White on Magenta
            "shattered_glass": (255, 255, 255),# White on Dark Green
            "flat_tire": (255, 255, 255),      # White on Orange-Red
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get coordinates
            x_min = int(bbox['x_min'])
            y_min = int(bbox['y_min'])
            x_max = int(bbox['x_max'])
            y_max = int(bbox['y_max'])
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))
            text_color = text_colors.get(class_name, (0, 0, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Calculate label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x_min, y_min - label_height - baseline - 5),
                (x_min + label_width, y_min),
                color,
                -1
            )
            
            # Draw label text with appropriate color
            cv2.putText(
                annotated,
                label,
                (x_min, y_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
        
        return annotated

    def calculate_metrics(self, detections: List[Dict]) -> Dict[str, int]:
        class_counts = {name: 0 for name in self.class_names.values()}
        total_pixels = 0

        for detection in detections:
            class_name = detection["class_name"]
            class_counts[class_name] += 1
            total_pixels += detection["pixel_area"]

        if total_pixels == 0:
            severity = "none"
        elif total_pixels < 5000:
            severity = "minor"
        elif total_pixels < 20000:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "total_detections": len(detections),
            "dents": class_counts["dent"],
            "scratches": class_counts["scratch"],
            "cracks": class_counts["crack"],
            "broken_lamps": class_counts["broken_lamp"],
            "shattered_glass": class_counts["shattered_glass"],
            "flat_tires": class_counts["flat_tire"],
            "severity": severity,
            "total_damage_pixels": total_pixels,
        }
