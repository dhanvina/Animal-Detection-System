import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from ultralytics import YOLO

model = None

class AnimalDetector:
    """
    Object-oriented animal detector using YOLOv8.
    Handles model loading, detection, and frame processing for images and videos.
    """
    def __init__(self, model_path: str = None, conf_threshold: float = 0.4, iou_threshold: float = 0.45):
        global model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        try:
            if model is None:
                print("Loading YOLO model...")
                if model_path and os.path.exists(model_path):
                    print(f"Loading custom model from {model_path}")
                    model = YOLO(model_path, verbose=False)
                else:
                    model = YOLO('yolov8x.pt', verbose=False)
                model.to('cpu')
                print("Using CPU for inference")
            self.model = model
            print("YOLO model loaded successfully!")
            print("Animal Detector initialized!")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise RuntimeError("Could not load YOLO model. Please check your internet connection and try again.")
        self.animal_categories = {
            'large_mammals': '🐘',
            'herbivores': '🦌',
            'carnivores': '🐺',
            'primates': '🐵',
            'birds': '🦉',
            'reptiles': '🐍',
            'small_mammals': '🐾'
        }
        self.animal_classes = {
            15: {'name': 'lion', 'category': 'large_mammals'},
            45: {'name': 'tiger', 'category': 'large_mammals'},
            16: {'name': 'leopard', 'category': 'large_mammals'},
            17: {'name': 'cheetah', 'category': 'carnivores'},
            20: {'name': 'elephant', 'category': 'large_mammals'},
            21: {'name': 'bear', 'category': 'large_mammals'},
            22: {'name': 'zebra', 'category': 'large_mammals'},
            23: {'name': 'giraffe', 'category': 'large_mammals'},
            18: {'name': 'buffalo', 'category': 'large_mammals'},
            46: {'name': 'rhino', 'category': 'large_mammals'},
            47: {'name': 'wildebeest', 'category': 'large_mammals'},
            19: {'name': 'hippopotamus', 'category': 'large_mammals'},
            0: {'name': 'deer', 'category': 'herbivores'},
            1: {'name': 'gazelle', 'category': 'herbivores'},
            2: {'name': 'antelope', 'category': 'herbivores'},
            3: {'name': 'springbok', 'category': 'herbivores'},
            4: {'name': 'oryx', 'category': 'herbivores'},
            5: {'name': 'sable_antelope', 'category': 'herbivores'},
            6: {'name': 'duiker', 'category': 'herbivores'},
            7: {'name': 'warthog', 'category': 'carnivores'},
            8: {'name': 'wild_boar', 'category': 'carnivores'},
            9: {'name': 'hyena', 'category': 'carnivores'},
            10: {'name': 'jackal', 'category': 'carnivores'},
            11: {'name': 'fox', 'category': 'carnivores'},
            13: {'name': 'pangolin', 'category': 'small_mammals'},
            14: {'name': 'baboon', 'category': 'primates'},
            24: {'name': 'monkey', 'category': 'primates'},
            25: {'name': 'aardvark', 'category': 'small_mammals'},
            26: {'name': 'porcupine', 'category': 'small_mammals'},
            27: {'name': 'ostrich', 'category': 'birds'},
            28: {'name': 'hornbill', 'category': 'birds'},
            29: {'name': 'secretary_bird', 'category': 'birds'},
            30: {'name': 'vulture', 'category': 'birds'},
            31: {'name': 'eagle', 'category': 'birds'},
            32: {'name': 'owl', 'category': 'birds'},
            33: {'name': 'guinea_fowl', 'category': 'birds'},
            34: {'name': 'crocodile', 'category': 'reptiles'},
            35: {'name': 'monitor_lizard', 'category': 'reptiles'},
            36: {'name': 'python', 'category': 'reptiles'},
            37: {'name': 'tortoise', 'category': 'reptiles'},
            38: {'name': 'civet', 'category': 'small_mammals'},
            39: {'name': 'genet', 'category': 'small_mammals'},
            40: {'name': 'mongoose', 'category': 'small_mammals'},
            41: {'name': 'badger', 'category': 'small_mammals'},
            42: {'name': 'hedgehog', 'category': 'small_mammals'},
            43: {'name': 'skunk', 'category': 'small_mammals'},
            44: {'name': 'bat', 'category': 'small_mammals'}
        }
        self.class_conf_thresholds = {
            'elephant': 0.6,
            'bear': 0.65,
            'big_cat': 0.55,
            'lion': 0.6,
            'tiger': 0.6,
            'leopard': 0.6,
            'rhino': 0.6,
            'hippopotamus': 0.6,
            'hyena': 0.5,
            'cheetah': 0.5,
            'fox': 0.5,
            'jackal': 0.5,
            'baboon': 0.45,
            'monkey': 0.45,
            'eagle': 0.5,
            'owl': 0.5,
            'vulture': 0.5,
            'crocodile': 0.55,
            'python': 0.5,
            'monitor_lizard': 0.45,
            'tortoise': 0.5,
            'default': conf_threshold
        }
    def _get_animal_class_ids(self) -> List[int]:
        return list(self.animal_classes.keys())
    def get_detection_message(self, class_identifier: Union[str, int]) -> str:
        if isinstance(class_identifier, int):
            if class_identifier in self.animal_classes:
                animal_info = self.animal_classes[class_identifier]
            else:
                return f"Detected: Unknown class {class_identifier}"
        elif isinstance(class_identifier, str):
            for class_id, info in self.animal_classes.items():
                if info['name'] == class_identifier:
                    animal_info = info
                    break
            else:
                return f"Detected: {class_identifier}"
        else:
            return "Detected: Unknown"
        category = animal_info['category']
        name = animal_info['name'].replace('_', ' ').title()
        emoji = self.animal_categories.get(category, '🐾')
        if category == 'large_mammals':
            return f"{emoji} WARNING: Large Mammal Detected - {name}! {emoji}"
        elif category == 'carnivores':
            return f"{emoji} Caution: {name} detected! {emoji}"
        else:
            return f"{emoji} Detected: {name} {emoji}"
    def _get_class_threshold(self, class_name: Union[str, int]) -> float:
        if isinstance(class_name, int) and class_name in self.animal_classes:
            class_name = self.animal_classes[class_name]['name']
        elif isinstance(class_name, dict) and 'name' in class_name:
            class_name = class_name['name']
        return self.class_conf_thresholds.get(str(class_name), self.class_conf_thresholds['default'])
    def detect_animals(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            target_size = 640
            scale = min(target_size / width, target_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            results = self.model(
                img_rgb, 
                imgsz=new_width if width > height else new_height,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self._get_animal_class_ids(),
                verbose=False
            )
            detections = []
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                conf = float(result.conf[0])
                class_id = int(result.cls[0])
                if class_id not in self.animal_classes:
                    continue
                class_name = self.animal_classes[class_id]
                class_threshold = self._get_class_threshold(class_name)
                if conf >= class_threshold:
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            return detections
        except Exception as e:
            print(f"Error in detect_animals: {str(e)}")
            raise
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > 640:
                scale = 640 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = width
                new_height = height
            results = self.model(
                frame_rgb,
                imgsz=new_width if width > height else new_height,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self._get_animal_class_ids(),
                verbose=False
            )
            detections = []
            categories_detected = set()
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                conf = float(result.conf[0])
                class_id = int(result.cls[0])
                if class_id not in self.animal_classes:
                    continue
                animal_info = self.animal_classes[class_id]
                class_name = animal_info['name']
                category = animal_info['category']
                class_threshold = self._get_class_threshold(class_name)
                if conf < class_threshold:
                    continue
                display_name = class_name.replace('_', ' ').title()
                detection = {
                    'class': class_name,
                    'display_name': display_name,
                    'category': category,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'alert': self.get_detection_message(class_id)
                }
                detections.append(detection)
                categories_detected.add(category)
                emoji = self.animal_categories.get(category, '🐾')
                label = f"{emoji} {display_name} {conf:.2f}"
                color = (0, 0, 255) if category == 'large_mammals' else \
                       (0, 165, 255) if category == 'carnivores' else \
                       (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset = 30
            if 'large_mammals' in categories_detected:
                alert_text = "🚨 WARNING: Large mammals detected!"
                cv2.putText(frame, alert_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30
            if 'carnivores' in categories_detected:
                alert_text = "⚠️ Caution: Carnivores detected!"
                cv2.putText(frame, alert_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame, detections
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame, []
    def process_video(self, video_path: str, output_path: str = None) -> str:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_detected{ext}"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, _ = self.process_frame(frame)
                out.write(processed_frame)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
        finally:
            cap.release()
            out.release()
        print(f"Video processing complete. Output saved to: {output_path}")
        return output_path
