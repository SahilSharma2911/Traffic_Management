import cv2
import numpy as np
from collections import defaultdict
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import time
import os
from tqdm import tqdm
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
        
        self.signal_states = defaultdict(lambda: {'red': 0, 'green': 0})

    def preprocess_frame(self, frame):
        """Convert frame to tensor for model input"""
        # Convert BGR to RGB
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        frame_tensor = F.to_tensor(frame)
        return frame_tensor.unsqueeze(0).to(self.device)

    def detect_vehicles(self, frame):
        """Detect vehicles in the frame"""
        frame_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            predictions = self.model(frame_tensor)
        
        # Get predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter detections based on confidence threshold
        confidence_threshold = 0.5
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Filter for vehicle classes only
        vehicle_mask = np.isin(labels, list(self.VEHICLE_CLASSES.keys()))
        boxes = boxes[vehicle_mask]
        labels = labels[vehicle_mask]
        scores = scores[vehicle_mask]
        
        return boxes, labels

    def analyze_traffic_density(self, boxes, labels, frame_dimensions):
        """Analyze traffic density and vehicle counts"""
        vehicle_count = defaultdict(int)
        
        for label in labels:
            if label in self.VEHICLE_CLASSES:
                vehicle_count[self.VEHICLE_CLASSES[label]] += 1
        
        # Calculate density (vehicles per unit area)
        frame_area = frame_dimensions[0] * frame_dimensions[1]
        density = len(boxes) / frame_area if frame_area > 0 else 0
        
        return vehicle_count, density

    def calculate_signal_timing(self, density, max_green_time=60, min_green_time=20):
        """Calculate optimal signal timing based on traffic density"""
        # Linear mapping of density to green time
        normalized_density = min(1.0, density * 1e5)  # Normalize density to 0-1 range
        green_time = min_green_time + (max_green_time - min_green_time) * normalized_density
        return float(np.clip(green_time, min_green_time, max_green_time))

    def draw_annotations(self, frame, boxes, labels, vehicle_count, signal_timing):
        """Draw bounding boxes and information on the frame with improved visualization"""
        # Draw bounding boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            vehicle_type = self.VEHICLE_CLASSES.get(int(label), 'unknown')
            
            # Different colors for different vehicle types
            color_map = {
                'car': (0, 255, 0),      # Green
                'bus': (255, 165, 0),    # Orange
                'truck': (255, 0, 0),    # Red
                'motorcycle': (255, 255, 0),  # Yellow
                'bicycle': (0, 255, 255)  # Cyan
            }
            color = color_map.get(vehicle_type, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with dark background for better visibility
            label_text = f"{vehicle_type}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Create semi-transparent overlay for information panel
        overlay = frame.copy()
        panel_height = 250  # Height of info panel
        panel_width = 300   # Width of info panel
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw traffic information with different colors
        info_text = [
            ("Total Vehicles:", sum(vehicle_count.values()), (255, 255, 255)),  # White
            ("Cars:", vehicle_count.get('car', 0), (0, 255, 0)),               # Green
            ("Buses:", vehicle_count.get('bus', 0), (255, 165, 0)),            # Orange
            ("Trucks:", vehicle_count.get('truck', 0), (255, 0, 0)),           # Red
            ("Motorcycles:", vehicle_count.get('motorcycle', 0), (255, 255, 0)), # Yellow
            ("Bicycles:", vehicle_count.get('bicycle', 0), (0, 255, 255)),     # Cyan
            ("Signal Timing:", f"{signal_timing:.1f}s", (147, 20, 255))        # Purple
        ]
        
        for i, (label, value, color) in enumerate(info_text):
            # Draw text with custom formatting
            text = f"{label} {value}"
            cv2.putText(frame, text, (20, 45 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def process_image(self, image_path):
        """Process a single image"""
        print(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get frame dimensions before any processing
        frame_dimensions = frame.shape[:2]  # (height, width)
        
        # Detect vehicles
        boxes, labels = self.detect_vehicles(frame)
        
        # Analyze traffic
        vehicle_count, density = self.analyze_traffic_density(boxes, labels, frame_dimensions)
        signal_timing = self.calculate_signal_timing(density)
        
        # Create annotated output
        annotated_frame = self.draw_annotations(frame.copy(), boxes, labels, vehicle_count, signal_timing)
        output_path = f"output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated_frame)
        
        # Create serializable result dictionary
        result = {
            'vehicle_count': dict(vehicle_count),
            'traffic_density': float(density),
            'recommended_green_time': float(signal_timing),
            'detection_boxes': boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            'output_path': output_path,
            'total_vehicles': sum(vehicle_count.values())
        }
        
        return result

def main():
    try:
        traffic_system = TrafficManagementSystem()
        
        # Process image
        try:
            image_path = "traffic.jpg"  # Make sure this image exists
            image_result = traffic_system.process_image(image_path)
            print("\nImage Analysis Results:")
            print(json.dumps(image_result, indent=2, cls=NumpyEncoder))
            print(f"\nAnnotated image saved as: {image_result['output_path']}")
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing image: {e}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()