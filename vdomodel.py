import cv2
import numpy as np
from collections import defaultdict
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import time
import os
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class TrafficManagementSystem:
    def __init__(self):
        print("Initializing Traffic Management System...")
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        self.VEHICLE_CLASSES = {
            3: 'car', 6: 'bus', 8: 'truck',
            4: 'motorcycle', 2: 'bicycle'
        }

    def preprocess_frame(self, frame):
        """Convert frame to tensor for model input"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame)
        return frame_tensor.unsqueeze(0).to(self.device)

    def detect_vehicles(self, frame):
        """Detect vehicles in the frame"""
        frame_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            predictions = self.model(frame_tensor)
        
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        confidence_threshold = 0.5
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        
        vehicle_mask = np.isin(labels, list(self.VEHICLE_CLASSES.keys()))
        boxes = boxes[vehicle_mask]
        labels = labels[vehicle_mask]
        
        return boxes, labels

    def analyze_traffic_density(self, boxes, labels, frame_dimensions):
        """Analyze traffic density and vehicle counts"""
        vehicle_count = defaultdict(int)
        
        for label in labels:
            if label in self.VEHICLE_CLASSES:
                vehicle_count[self.VEHICLE_CLASSES[label]] += 1
        
        frame_area = frame_dimensions[0] * frame_dimensions[1]
        density = len(boxes) / frame_area if frame_area > 0 else 0
        
        return vehicle_count, density

    def calculate_signal_timing(self, density, max_green_time=60, min_green_time=20):
        """Calculate optimal signal timing based on traffic density"""
        normalized_density = min(1.0, density * 1e5)
        green_time = min_green_time + (max_green_time - min_green_time) * normalized_density
        return float(np.clip(green_time, min_green_time, max_green_time))

    def draw_annotations(self, frame, boxes, labels, vehicle_count, signal_timing):
        """Draw bounding boxes and information on the frame"""
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            vehicle_type = self.VEHICLE_CLASSES.get(int(label), 'unknown')
            
            color_map = {
                'car': (0, 255, 0),      # Green
                'bus': (255, 165, 0),    # Orange
                'truck': (255, 0, 0),    # Red
                'motorcycle': (255, 255, 0),  # Yellow
                'bicycle': (0, 255, 255)  # Cyan
            }
            color = color_map.get(vehicle_type, (0, 255, 0))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{vehicle_type}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        overlay = frame.copy()
        panel_height = 250  
        panel_width = 300   
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        info_text = [
            ("Total Vehicles:", sum(vehicle_count.values()), (255, 255, 255)),
            ("Cars:", vehicle_count.get('car', 0), (0, 255, 0)),
            ("Buses:", vehicle_count.get('bus', 0), (255, 165, 0)),
            ("Trucks:", vehicle_count.get('truck', 0), (255, 0, 0)),
            ("Motorcycles:", vehicle_count.get('motorcycle', 0), (255, 255, 0)),
            ("Bicycles:", vehicle_count.get('bicycle', 0), (0, 255, 255)),
            ("Signal Timing:", f"{signal_timing:.1f}s", (147, 20, 255))
        ]
        
        for i, (label, value, color) in enumerate(info_text):
            text = f"{label} {value}"
            cv2.putText(frame, text, (20, 45 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame

    def process_video_realtime(self, video_path):
        """Process video in real-time and display annotated frames"""
        print(f"Processing video in real-time: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_dimensions = frame.shape[:2]
                boxes, labels = self.detect_vehicles(frame)
                vehicle_count, density = self.analyze_traffic_density(boxes, labels, frame_dimensions)
                signal_timing = self.calculate_signal_timing(density)
                
                annotated_frame = self.draw_annotations(frame.copy(), boxes, labels, vehicle_count, signal_timing)
                
                cv2.imshow('Traffic Management System - Real-Time Analysis', annotated_frame)
                
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\nTotal frames processed: {frame_count}")

def main():
    traffic_system = TrafficManagementSystem()
    video_path = "road.mp4"
    traffic_system.process_video_realtime(video_path)

if __name__ == "__main__":
    main()
