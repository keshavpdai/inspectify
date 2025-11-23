"""
Post-processing filters for damage detection
Reduces false positives specific to certain damage types
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DamageFilter:
    """Post-processing filters to reduce false positives"""

    def __init__(self):
        self.filters = {
            'shattered_glass': self.filter_glass_damage,
            'broken_lamp': self.filter_glass_damage,  # Water can be misclassified as broken_lamp too
            'dent': self.filter_dent,
            'scratch': self.filter_scratch
        }

    def apply_filters(
        self, 
        detections: List[Dict], 
        image: np.ndarray,
        enable_filters: Dict[str, bool] = None
    ) -> List[Dict]:
        """
        Apply post-processing filters to detections

        Args:
            detections: List of detection dicts from YOLO
            image: Original image (for analysis)
            enable_filters: Which filters to apply (default: all)

        Returns:
            Filtered detection list
        """
        if enable_filters is None:
            enable_filters = {
                'shattered_glass': True,
                'broken_lamp': True,  # Also filter broken_lamp for water droplets
                'dent': False,  # Two-stage detection handles this
                'scratch': False
            }

        logger.info(f"🔍 Filtering {len(detections)} detections with filters: {enable_filters}")
        filtered = []

        for i, detection in enumerate(detections):
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            logger.info(f"  Detection {i+1}: {class_name} (conf: {confidence:.2f})")

            # Apply class-specific filter if enabled
            if class_name in enable_filters and enable_filters[class_name]:
                should_keep = self._should_keep_detection(detection, image, class_name)
                if should_keep:
                    filtered.append(detection)
                    logger.info(f"    ✅ KEPT {class_name}")
                else:
                    logger.warning(f"    ❌ FILTERED OUT {class_name} with conf {confidence:.2f}")
            else:
                # No filter for this class, keep it
                filtered.append(detection)
                logger.info(f"    ✅ KEPT {class_name} (no filter)")

        logger.info(f"📊 Filter results: {len(detections)} → {len(filtered)} detections")
        return filtered

    def _should_keep_detection(
        self, 
        detection: Dict, 
        image: np.ndarray, 
        class_name: str
    ) -> bool:
        """Run class-specific filter"""
        filter_func = self.filters.get(class_name)
        if filter_func:
            return filter_func(detection, image)
        return True

    def filter_glass_damage(
        self, 
        detection: Dict, 
        image: np.ndarray
    ) -> bool:
        """
        Filter glass-related false positives (water droplets, reflections)
        
        Works for both shattered_glass and broken_lamp since water droplets
        can be misclassified as either.

        Strategy:
        1. Extract detected region
        2. Check for water droplet patterns vs actual glass cracks
        3. Analyze texture variance
        4. Apply heuristics

        Returns:
            True = keep detection, False = filter out
        """
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']

        logger.info(f"      🔬 Analyzing {class_name} region...")

        # Extract region
        x_min = int(bbox['x_min'])
        y_min = int(bbox['y_min'])
        x_max = int(bbox['x_max'])
        y_max = int(bbox['y_max'])

        # Ensure valid coordinates
        h, w = image.shape[:2]
        x_min = max(0, min(x_min, w-1))
        y_min = max(0, min(y_min, h-1))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            logger.info(f"      ❌ Invalid region coordinates")
            return False  # Invalid region

        region = image[y_min:y_max, x_min:x_max]

        if region.size == 0:
            logger.info(f"      ❌ Empty region")
            return False

        # HEURISTIC 1: Check for high-frequency patterns (water droplets)
        # Water creates many small bright spots, cracks create linear patterns
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Detect bright spots (water droplets reflect light)
        # Lower threshold to catch water on darker surfaces
        _, bright_spots = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_spots > 0) / bright_spots.size

        logger.info(f"      💡 Bright ratio: {bright_ratio:.2%}")
        
        if bright_ratio > 0.12:  # >12% of region is bright (lowered from 15%)
            logger.info(f"      ❌ FILTER: Likely water droplets (bright ratio: {bright_ratio:.2%})")
            return False  # Likely water droplets

        # HEURISTIC 2: Check for edge patterns
        # Real shattered glass has radial crack patterns
        # Water creates random scattered patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        logger.info(f"      📐 Edge density: {edge_density:.2%}")

        if edge_density < 0.05:  # Very few edges
            logger.info(f"      ❌ FILTER: Too few edges for glass damage (density: {edge_density:.2%})")
            return False

        # HEURISTIC 3: Confidence adjustment for ambiguous cases
        # If confidence is high (>0.85) but patterns are suspicious, still filter
        if confidence > 0.85 and (bright_ratio > 0.08 or edge_density < 0.10):
            logger.info(f"      ❌ FILTER: High confidence ({confidence:.2f}) but suspicious patterns")
            return False

        # HEURISTIC 4: Check for blur (water creates blur, shattered glass is sharp)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        logger.info(f"      🔍 Laplacian variance: {laplacian_var:.1f}")

        if laplacian_var < 100:  # Increased from 50 to be more aggressive
            logger.info(f"      ❌ FILTER: Blurry region (laplacian var: {laplacian_var:.1f})")
            return False
        
        # HEURISTIC 5: Check for scattered pattern (water droplets)
        # Water creates many small disconnected regions
        # Real glass damage is more continuous
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Count small components (water droplets are small and scattered)
        small_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] < 100)
        
        logger.info(f"      🔬 Small components: {small_components}")
        
        if small_components > 10:  # Many small scattered regions
            logger.info(f"      ❌ FILTER: Scattered pattern (small components: {small_components})")
            return False

        # Passed all filters, keep detection
        logger.info(f"      ✅ PASS: All heuristics passed, keeping detection")
        return True

    def filter_dent(self, detection: Dict, image: np.ndarray) -> bool:
        """
        Filter dent false positives
        Note: Most dent FPs should be handled by two-stage detection
        This is a backup for edge cases
        """
        confidence = detection['confidence']

        # Only apply for low-confidence dents (<0.6)
        if confidence < 0.6:
            bbox = detection['bbox']
            x_min = int(bbox['x_min'])
            y_min = int(bbox['y_min'])
            x_max = int(bbox['x_max'])
            y_max = int(bbox['y_max'])

            h, w = image.shape[:2]
            x_min = max(0, min(x_min, w-1))
            y_min = max(0, min(y_min, h-1))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))

            if x_max <= x_min or y_max <= y_min:
                return False

            region = image[y_min:y_max, x_min:x_max]

            if region.size == 0:
                return False

            # Check if region has consistent texture (likely surface, not dent)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            texture_var = np.var(gray)

            if texture_var > 1000:  # High variance = textured surface
                logger.info(f"Textured surface, not dent (var: {texture_var:.1f})")
                return False

        return True

    def filter_scratch(self, detection: Dict, image: np.ndarray) -> bool:
        """Filter scratch false positives (optional)"""
        # Currently no specific filtering
        # Two-stage detection should handle most scratch FPs
        return True


# Convenience function for easy integration
def filter_detections(
    detections: List[Dict],
    image: np.ndarray,
    enable_shattered_glass_filter: bool = True
) -> List[Dict]:
    """
    Quick filter function for use in main.py

    Usage:
        detections = detector.predict(image, conf=0.5)
        filtered = filter_detections(detections, image)
    """
    filters = DamageFilter()

    return filters.apply_filters(
        detections,
        image,
        enable_filters={
            'shattered_glass': enable_shattered_glass_filter,
            'broken_lamp': enable_shattered_glass_filter,  # Use same filter for broken_lamp
            'dent': False,
            'scratch': False
        }
    )
