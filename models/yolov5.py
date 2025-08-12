import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class YOLOv5BallDetector:
    """YOLOv5 ONNX Ball Detector for table tennis ball detection"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.45):
        """
        Initialize the YOLOv5 ball detector
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for filtering overlapping boxes
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input size: {self.input_width}x{self.input_height}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess the input image for YOLO inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor, scale_x, scale_y
        """
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factors
        scale_x = self.input_width / original_width
        scale_y = self.input_height / original_height
        
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale_x, scale_y
    
    def postprocess(self, outputs: np.ndarray, scale_x: float, scale_y: float) -> List[Tuple[int, int, int, int, float]]:
        """
        Postprocess YOLO outputs to get bounding boxes
        
        Args:
            outputs: Raw model outputs
            scale_x: X scaling factor
            scale_y: Y scaling factor
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # YOLOv5 output format: [batch_size, num_predictions, 85] or [batch_size, num_predictions, 6]
        # Each prediction: [cx, cy, w, h, conf, class_probs...]
        predictions = outputs[0]  # Shape: [num_predictions, 85] or [num_predictions, 6]
        
        # Handle different output shapes
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        boxes = []
        confidences = []
        
        for prediction in predictions:
            # Extract box coordinates and confidence
            if len(prediction) >= 5:
                cx, cy, w, h, confidence = prediction[:5]
                
                # Handle potential class scores (take max if multiple classes)
                if len(prediction) > 5:
                    class_scores = prediction[5:]
                    if len(class_scores) > 0:
                        confidence = confidence * np.max(class_scores)
                
                # Convert to scalar if it's an array
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                
                # Filter by confidence threshold
                if confidence >= self.conf_threshold:
                    # Convert center coordinates to corner coordinates
                    x1 = int((cx - w/2) / scale_x)
                    y1 = int((cy - h/2) / scale_y)
                    x2 = int((cx + w/2) / scale_x)
                    y2 = int((cy + h/2) / scale_y)
                    
                    boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x, y, w, h for NMS
                    confidences.append(float(confidence))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            final_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    final_detections.append((x, y, x + w, y + h, conf))
            
            return final_detections
        
        return []
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect balls in the input image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # Preprocess image
        input_tensor, scale_x, scale_y = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Debug: Print output shape on first detection
        if not hasattr(self, '_debug_printed'):
            print(f"Model output shape: {[out.shape for out in outputs]}")
            print(f"First output shape: {outputs[0].shape}")
            if len(outputs[0].shape) == 3:
                print(f"Sample predictions shape: {outputs[0][0][:5].shape if outputs[0].shape[1] > 5 else outputs[0][0].shape}")
            self._debug_printed = True
        
        # Postprocess results
        detections = self.postprocess(outputs, scale_x, scale_y)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw bounding boxes on the image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f'Ball: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image