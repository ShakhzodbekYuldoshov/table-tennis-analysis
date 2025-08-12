import cv2
import numpy as np
import logging
from collections import deque
from typing import List, Tuple, Optional
import math
import time

logging.basicConfig(level=logging.INFO, filename='track_object.log',)
logger = logging.getLogger(__name__)



class TrackObject:
    def __init__(self, contour, frame_number, object_id: int):
        self.object_id = object_id
        self.contour = contour
        self.first_time_appeared = frame_number
        self.last_time_appeared = frame_number
        self.centers_history = []  # List of (x, y) center positions
        self.movement_distances = []  # List of distances between consecutive frames
        self.total_movement = 0.0
        self.is_active = True
        self.frames_since_last_seen = 0
        
        # Calculate and store initial center
        center = self._calculate_center(contour)
        self.x, self.y, self.width, self.height = cv2.boundingRect(contour)
        self.centers_history.append(center)
        self.current_center = center
    
    def _calculate_center(self, contour) -> Tuple[float, float]:
        """Calculate center point of contour"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # Fallback to bounding rect center
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w/2, y + h/2
        return (cx, cy)
    
    def update(self, contour, frame_number):
        """Update object with new contour detection"""
        self.contour = contour
        self.last_time_appeared = frame_number
        self.frames_since_last_seen = 0
        
        # Calculate new center
        new_center = self._calculate_center(contour)
        
        # Calculate movement distance from previous center
        if len(self.centers_history) > 0:
            prev_center = self.centers_history[-1]
            distance = math.sqrt((new_center[0] - prev_center[0])**2 + 
                               (new_center[1] - prev_center[1])**2)
            self.movement_distances.append(distance)
            self.total_movement += distance
        
        # Update history
        self.centers_history.append(new_center)
        self.current_center = new_center
        
        # Keep only last 20 frames of history
        if len(self.centers_history) > 20:
            self.centers_history.pop(0)
            if len(self.movement_distances) > 19:  # 19 distances for 20 positions
                removed_distance = self.movement_distances.pop(0)
                self.total_movement -= removed_distance
    
    def increment_frames_since_seen(self):
        """Increment counter for frames where object wasn't detected"""
        self.frames_since_last_seen += 1
        if self.frames_since_last_seen > 5:  # Mark inactive after 5 frames
            self.is_active = False
    
    def get_average_movement(self) -> float:
        """Get average movement per frame over tracking history"""
        if len(self.movement_distances) == 0:
            return 0.0
        return sum(self.movement_distances) / len(self.movement_distances)
    
    def get_movement_variance(self) -> float:
        """Calculate variance in movement to detect consistent motion patterns"""
        if len(self.movement_distances) < 2:
            return 0.0
        
        avg_movement = self.get_average_movement()
        variance = sum((d - avg_movement)**2 for d in self.movement_distances)
        return variance / len(self.movement_distances)
    
    def get_tracking_quality_score(self) -> float:
        """Calculate quality score based on movement patterns (higher = more likely to be ball)"""
        if len(self.movement_distances) < 3:
            return 0.0
        
        # Ball characteristics:
        # 1. Consistent significant movement (not too low, not too erratic)
        # 2. Reasonable movement variance (smooth trajectory)
        # 3. Sufficient tracking history
        
        avg_movement = self.get_average_movement()
        movement_variance = self.get_movement_variance()
        tracking_frames = len(self.movement_distances)
        
        # Score components
        movement_score = min(avg_movement / 50.0, 1.0) if avg_movement > 5 else 0  # Prefer moderate movement
        consistency_score = 1.0 / (1.0 + movement_variance / 100.0)  # Lower variance = higher score
        history_score = min(tracking_frames / 10.0, 1.0)  # Prefer longer tracking history
        
        return (movement_score * 0.4 + consistency_score * 0.4 + history_score * 0.2)


class BallTracker:
    def __init__(self, max_distance_threshold: float = 400.0, min_tracking_frames: int = 10):
        self.tracked_objects = []  # List of TrackObject instances
        self.next_object_id = 0
        self.max_distance_threshold = max_distance_threshold
        self.min_tracking_frames = min_tracking_frames
        self.current_frame = 0
        self.ball_candidate = None  # The most likely ball object
        self.bouncing_points = deque(maxlen=3)  # Store bouncing points for ball trajectory
        
        # FPS tracking
        self.frame_times = []
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.last_frame_time = time.time()  # Track time of last frame
    
    def update(self, contours: List, frame_number: int):
        """Update tracker with new frame detections"""
        current_time = time.time()
        self.current_frame = frame_number

        if len(contours) == 0:
            # No detections, increment frames_since_seen for all objects
            for obj in self.tracked_objects:
                obj.increment_frames_since_seen()
            self._update_fps(current_time)
            return
        
        # Calculate centers for current detections
        current_centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w/2, y + h/2
            current_centers.append((cx, cy))
        
        # Match current detections to existing tracked objects
        used_detections = set()
        used_objects = set()
        
        # Find best matches using Hungarian-like approach (simplified)
        for obj in self.tracked_objects:
            if not obj.is_active:
                continue
                
            best_match_idx = -1
            best_distance = float('inf')
            best_distance_center = None
            close_matches = []
            for i, center in enumerate(current_centers):
                if i in used_detections:
                    continue
                    
                distance = math.sqrt((center[0] - obj.current_center[0])**2 + 
                                   (center[1] - obj.current_center[1])**2)
                
                logger.info(f"Distance from object {obj.object_id} to detection {i}: {distance:.2f} at center {center}")
                if distance < self.max_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
                    best_distance_center = center

                if best_distance != distance and abs(best_distance - distance) < 40:  # Consider close matches
                    close_matches.append({
                        "match_index": i,
                        "distance": distance,
                        "center": center
                    })

            close_matches.append({
                "match_index": best_match_idx,
                "distance": best_distance,
                "center": best_distance_center
            })

            if len(close_matches) > 1:
                logger.info(f"frame: {frame_number}")
                logger.info(f"Close matches for object {obj.object_id}: {close_matches}")
                # Resolve close matches (e.g., by selecting the one with lowest x point of center)
                best_match = min(close_matches, key=lambda x: x["center"][1])
                logger.info(f"Selected close match for object {obj.object_id}: {best_match}")
                best_match_idx = best_match["match_index"]
                best_distance = best_match["center"]

            if best_match_idx != -1:
                # Update existing object
                obj.update(contours[best_match_idx], frame_number)
                used_detections.add(best_match_idx)
                used_objects.add(obj.object_id)
            else:
                # Object not found in current frame
                obj.increment_frames_since_seen()
        
        # Create new tracked objects for unmatched detections
        for i, contour in enumerate(contours):
            if i not in used_detections:
                new_obj = TrackObject(contour, frame_number, self.next_object_id)
                self.tracked_objects.append(new_obj)
                self.next_object_id += 1
        
        # Remove inactive objects
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.is_active]
        
        # Update ball candidate after sufficient tracking
        if frame_number >= self.min_tracking_frames:
            self._update_ball_candidate()
        
        # Update FPS tracking
        self._update_fps(current_time)
    
    def _update_ball_candidate(self):
        """Identify the most likely ball candidate based on tracking analysis"""
        if not self.tracked_objects:
            self.ball_candidate = None
            return
        
        # Filter objects with minimum tracking history
        eligible_objects = [obj for obj in self.tracked_objects 
                           if len(obj.movement_distances) >= 3]
        
        if not eligible_objects:
            self.ball_candidate = None
            return
        
        # Prioritize by tracking history (frames tracked) first, then by quality score
        best_object = None
        best_tracking_frames = 0
        best_score = 0.0
        
        for obj in eligible_objects:
            tracking_frames = len(obj.centers_history)
            
            # Primary criteria: longest tracking history
            # Secondary criteria: highest quality score (tie breaker)
            if (tracking_frames > best_tracking_frames or 
                (tracking_frames == best_tracking_frames)):
                best_tracking_frames = tracking_frames
                best_object = obj
        
        # Only one ball candidate at a time
        self.ball_candidate = best_object
    
    def _update_fps(self, current_frame_time):
        """Update FPS calculation and logger.info every second"""
        # Calculate time since last frame (actual FPS)
        if hasattr(self, 'last_frame_time'):
            frame_interval = current_frame_time - self.last_frame_time
            self.frame_times.append(frame_interval)
        
        self.last_frame_time = current_frame_time
        
        # Keep only recent frame times (last 30 frames for smooth averaging)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Update FPS every second
        if current_frame_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.frame_times) > 0:
                avg_frame_interval = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_interval if avg_frame_interval > 0 else 0.0
                logger.info(f"FPS: {self.current_fps:.1f}")
            
            self.last_fps_update = current_frame_time
        
    def draw_tracking_info(self, image):
        """Draw tracking information on image"""
        # Draw FPS on top-left corner
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(image, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for obj in self.tracked_objects:
            if not obj.is_active:
                continue
            print("Drawing tracking info on image")
            
            # Different visualization for ball candidate vs other objects
            is_ball = (self.ball_candidate and obj.object_id == self.ball_candidate.object_id)

            if is_ball:
                # Ball candidate - highlighted in red
                cv2.drawContours(image, [obj.contour], -1, (0, 0, 255), 3)  # Red contour
                center = (int(obj.current_center[0]), int(obj.current_center[1]))
                cv2.circle(image, center, 8, (0, 0, 255), 3)  # Red circle
                
                # Draw prominent ball label
                cv2.putText(image, "BALL", (center[0] + 15, center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                
                # Draw extended movement trail for ball
                if len(obj.centers_history) > 1:
                    points = np.array([(int(x), int(y)) for x, y in obj.centers_history], np.int32)
                    cv2.polylines(image, [points], False, (0, 0, 255), 2)  # Red trail
                
                # check whether it is going up or down
                if len(obj.centers_history) > 1:
                    last_center = obj.centers_history[-1]
                    second_last_center = obj.centers_history[-2]
                    third_last_center = obj.centers_history[-3] if len(obj.centers_history) > 2 else second_last_center

                    if last_center[1] < second_last_center[1]:
                        direction_text = "UP"

                        # check whether its direction changed or not by x values
                        is_direction_changed = last_center[0] < second_last_center[0] and third_last_center[0] > second_last_center[0] or last_center[0] > second_last_center[0] and third_last_center[0] < second_last_center[0]
                        is_bouncing = third_last_center[1] < second_last_center[1] and is_direction_changed

                        if is_bouncing:
                            direction_text = "BOUNCE"
                            print(f"Height: {obj.height}, Center: {obj.current_center}")
                            self.bouncing_points.append((second_last_center[0], second_last_center[1] + int(obj.height / 2)))
                    else:
                        direction_text = "DOWN"
                    cv2.putText(image, direction_text, (center[0] + 15, center[1] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Ball detailed info
                ball_info = f"ID:{obj.object_id} Frames:{len(obj.centers_history)} Avg:{obj.get_average_movement():.1f}"
                cv2.putText(image, ball_info, (center[0] + 15, center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            else:
                # Other objects - dimmed visualization
                cv2.drawContours(image, [obj.contour], -1, (100, 100, 100), 1)  # Gray contour
                center = (int(obj.current_center[0]), int(obj.current_center[1]))
                cv2.circle(image, center, 3, (100, 100, 100), -1)  # Gray dot
                
                # Dimmed movement trail
                if len(obj.centers_history) > 1:
                    points = np.array([(int(x), int(y)) for x, y in obj.centers_history], np.int32)
                    cv2.polylines(image, [points], False, (50, 50, 50), 1)  # Dark gray trail
                
                # Basic object info (smaller text)
                info_text = f"ID:{obj.object_id}"
                cv2.putText(image, info_text, (center[0] + 5, center[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw bouncing points if available
        if self.bouncing_points:
            for point in self.bouncing_points:
                cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        
        return image
