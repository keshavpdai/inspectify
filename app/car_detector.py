"""
Car Detection Module for Inspectify
Stage 1: Detect vehicles before damage detection
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CarDetector:
    """Stage 1: Detect vehicles using YOLOv8 pretrained on COCO"""
    
    def __init__(self, model_path: Union[str, Path] = "models/yolov8n.pt", device: str = None):
        """
        Initialize car detector
        
        Args:
            model_path: Path to YOLO model (auto-downloads if not exists)
                       Default: models/yolov8n.pt (~6MB)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert Path to string if needed
        model_path_str = str(model_path) if isinstance(model_path, Path) else model_path
        
        # YOLO will auto-download to the specified path if it doesn't exist
        self.model = YOLO(model_path_str)
        self.model.to(self.device)
        
        # COCO vehicle classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        logger.info(f"✅ Car detector loaded: {model_path_str} on {self.device}")
    
    def detect_vehicles(
        self, 
        image: np.ndarray, 
        conf: float = 0.4,
        include_classes: List[str] = ['car']
    ) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect vehicles in image
        
        Args:
            image: Input image (numpy array, BGR format)
            conf: Confidence threshold (0.0-1.0)
            include_classes: Vehicle types to detect ['car', 'truck', 'bus']
            
        Returns:
            List of (x_min, y_min, x_max, y_max, confidence, class_name)
        """
        results = self.model(image, conf=conf, verbose=False)
        
        if not results:
            logger.info("No detection results returned")
            return []
        
        result = results[0]
        vehicles = []
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names.get(class_id, 'unknown')
            
            # Filter for specified vehicle classes only
            if class_name in include_classes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                vehicles.append((
                    int(x_min), int(y_min), 
                    int(x_max), int(y_max),
                    confidence,
                    class_name
                ))
        
        logger.info(f"🚗 Detected {len(vehicles)} vehicle(s) in image")
        return vehicles
    
    def crop_vehicle_regions(
        self, 
        image: np.ndarray, 
        vehicles: List[Tuple[int, int, int, int, float, str]],
        padding: int = 20
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract vehicle regions from image with padding
        
        Args:
            image: Original full image
            vehicles: List of vehicle detections from detect_vehicles()
            padding: Extra pixels around bounding box (default 20)
            
        Returns:
            List of (cropped_image, (x_min, y_min, x_max, y_max))
        """
        h, w = image.shape[:2]
        cropped_regions = []
        
        for (x_min, y_min, x_max, y_max, conf, cls) in vehicles:
            # Add padding while staying within image bounds
            x_min_pad = max(0, x_min - padding)
            y_min_pad = max(0, y_min - padding)
            x_max_pad = min(w, x_max + padding)
            y_max_pad = min(h, y_max + padding)
            
            # Crop region
            cropped = image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            
            cropped_regions.append((
                cropped,
                (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
            ))
            
            logger.debug(f"Cropped vehicle region: {cropped.shape} at ({x_min_pad}, {y_min_pad})")
        
        return cropped_regions
    
    def get_largest_vehicle(
        self,
        vehicles: List[Tuple[int, int, int, int, float, str]]
    ) -> Optional[Tuple[int, int, int, int, float, str]]:
        """
        Get the largest vehicle (by area) from detections
        Useful when you want to focus on main vehicle only
        
        Returns:
            Single vehicle tuple or None if no vehicles
        """
        if not vehicles:
            return None
        
        largest = max(vehicles, key=lambda v: (v[2] - v[0]) * (v[3] - v[1]))
        return largest
