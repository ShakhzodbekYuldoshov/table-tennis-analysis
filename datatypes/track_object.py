import cv2
import time
import math
import numpy as np
import logging
from collections import deque
from typing import List, Tuple, Optional

from datatypes.events import BallOutOfBoundsEvent, \
                             EventManager, \
                             TableSide, \
                             BallTableContactEvent, \
                             BallNetContactEvent, \
                             EventType

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
    
    def is_direction_changed(self) -> bool:
        """Check if direction changed based on last 3 centers"""
        if len(self.centers_history) < 3:
            return False
        
        last = self.centers_history[-1]
        second_last = self.centers_history[-2]
        third_last = self.centers_history[-3]
        
        # Check if x direction changed
        return (last[0] < second_last[0] and third_last[0] > second_last[0]) or \
               (last[0] > second_last[0] and third_last[0] < second_last[0])


class BallTracker:
    def __init__(self, max_distance_threshold: float = 400.0, min_tracking_frames: int = 10, event_manager: Optional[EventManager] = None):
        self.tracked_objects = []  # List of TrackObject instances
        self.next_object_id = 0
        self.max_distance_threshold = max_distance_threshold
        self.min_tracking_frames = min_tracking_frames
        self.current_frame = 0
        self.ball_candidate = None  # The most likely ball object
        self.bouncing_points = deque(maxlen=3)  # Store bouncing points for ball trajectory
        self.event_manager = event_manager
        
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
        
    def draw_tracking_info(self, image, table_left_polygon, table_right_polygon, frame_number: int, net_polygon=None):
        """Draw tracking information on image"""
        # Draw FPS on top-left corner
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(image, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for obj in self.tracked_objects:
            if not obj.is_active:
                continue
            
            # Different visualization for ball candidate vs other objects
            is_ball = (self.ball_candidate and obj.object_id == self.ball_candidate.object_id)

            if is_ball:
                # Check for net contact if net polygon is provided
                ball_in_net = False
                if net_polygon and self.event_manager:
                    ball_in_net = self.point_within_polygon(obj.current_center, net_polygon)
                    self.check_net_contact(obj, net_polygon, frame_number)
                
                # Ball candidate - highlighted in red (or orange if in net area)
                ball_color = (0, 165, 255) if ball_in_net else (0, 0, 255)  # Orange if in net, red otherwise
                cv2.drawContours(image, [obj.contour], -1, ball_color, 3)
                center = (int(obj.current_center[0]), int(obj.current_center[1]))
                cv2.circle(image, center, 8, ball_color, 3)
                
                # Draw prominent ball label with net status
                ball_label = "BALL (IN NET)" if ball_in_net else "BALL"
                cv2.putText(image, ball_label, (center[0] + 15, center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 3)
                
                # Draw extended movement trail for ball
                if len(obj.centers_history) > 1:
                    points = np.array([(int(x), int(y)) for x, y in obj.centers_history], np.int32)
                    cv2.polylines(image, [points], False, ball_color, 2)
                
                    
                # Determine ball direction and status
                if len(obj.centers_history) > 2:
                    last_center = obj.centers_history[-1]
                    second_last_center = obj.centers_history[-2]
                    third_last_center = obj.centers_history[-3]

                if last_center[1] < second_last_center[1]:
                    direction_text = "UP"

                    # check whether its direction changed or not by x values
                    is_direction_changed = obj.is_direction_changed()
                    is_bouncing = third_last_center[1] < second_last_center[1] and is_direction_changed

                    if is_bouncing:
                        direction_text = "BOUNCE"
                        print(f"Height: {obj.height}, Center: {obj.current_center}")
                        object_bottom = second_last_center[0], second_last_center[1] + int(obj.height / 2)
                        on_left_table = self.point_within_polygon(object_bottom, table_left_polygon)
                        on_right_table = self.point_within_polygon(object_bottom, table_right_polygon)
                        if on_left_table or on_right_table:
                            direction_text = "BOUNCE"
                            self.bouncing_points.append((object_bottom))
                            self.event_manager.add_event(
                                BallTableContactEvent(
                                    timestamp=frame_number-1,
                                    frame_number=frame_number-1,
                                    contact_position=object_bottom,
                                    table_side=TableSide.RIGHT if on_right_table else TableSide.LEFT,
                                    bounce_height=2
                                )
                            )
                else:
                    direction_text = "DOWN"
                cv2.putText(image, direction_text, (center[0] + 15, center[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Ball detailed info
                ball_info = f"ID:{obj.object_id} Frames:{len(obj.centers_history)} Avg:{obj.get_average_movement():.1f}"
                cv2.putText(image, ball_info, (center[0] + 15, center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 2)
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

        # Draw texts for events
        start_y = 50
        for i, event in enumerate(self.event_manager.get_events()):
            if isinstance(event, BallTableContactEvent):
                cv2.putText(image, f"Bounce: {event.table_side.name}", 
                            (10, start_y + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif isinstance(event, BallNetContactEvent):
                cv2.putText(image, "Net Contact", 
                            (10, start_y + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif isinstance(event, BallOutOfBoundsEvent):
                cv2.putText(image, "Out of Bounds", 
                            (10, start_y + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image
    
    def point_within_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is within a polygon using ray-casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def check_net_contact(self, ball_obj: TrackObject, net_polygon: List[Tuple[float, float]], frame_number: int) -> bool:
        """Check if ball has contacted the net (bounced/changed direction in net area) and trigger event if so"""
        if len(ball_obj.centers_history) < 4:  # Need at least 4 points to detect direction change
            return False
        
        current_center = ball_obj.centers_history[-1]
        previous_center = ball_obj.centers_history[-2]
        third_last_center = ball_obj.centers_history[-3]
        fourth_last_center = ball_obj.centers_history[-4]
        
        # Check if ball is currently in net area or was in net area recently
        current_in_net = self.point_within_polygon(current_center, net_polygon)
        previous_in_net = self.point_within_polygon(previous_center, net_polygon)
        third_last_in_net = self.point_within_polygon(third_last_center, net_polygon)
        
        # Ball must be in or have been in the net area
        if not (current_in_net or previous_in_net or third_last_in_net):
            return False
        
        # Calculate velocity vectors for direction change detection
        # Vector from 4th to 3rd point
        vel1 = (third_last_center[0] - fourth_last_center[0], 
                third_last_center[1] - fourth_last_center[1])
        
        # Vector from 3rd to 2nd point  
        vel2 = (previous_center[0] - third_last_center[0], 
                previous_center[1] - third_last_center[1])
        
        # Vector from 2nd to current point
        vel3 = (current_center[0] - previous_center[0], 
                current_center[1] - previous_center[1])
        
        # Check for significant direction change in X direction (horizontal bounce)
        # This indicates the ball hit the net and bounced back
        x_direction_change = self._detect_x_direction_change(vel1, vel2, vel3)
        
        # Check for significant velocity reduction (ball slowed down after hitting net)
        velocity_reduction = self._detect_velocity_reduction(vel1, vel2, vel3)
        
        # Check for upward trajectory change (ball hit net and went up)
        upward_bounce = self._detect_upward_bounce(vel1, vel2, vel3)
        
        # Net contact detected if any of these conditions are met while in net area
        net_contact_detected = (x_direction_change or velocity_reduction or upward_bounce)
        
        if net_contact_detected and (current_in_net or previous_in_net):
            # Use the point where ball was closest to center of net polygon
            contact_position = self._find_closest_point_to_net_center(
                [fourth_last_center, third_last_center, previous_center, current_center], 
                net_polygon
            )
            
            # Calculate impact velocity (velocity just before contact)
            impact_velocity = vel2  # Velocity entering the net area
            
            # Add net contact event
            if self.event_manager:
                # Check if we already detected a net contact recently (avoid duplicates)
                recent_net_events = [e for e in self.event_manager.get_events_by_type(EventType.BALL_NET_CONTACT) 
                                   if abs(e.frame_number - frame_number) < 5]
                
                if not recent_net_events:  # Only add if no recent net contact detected
                    self.event_manager.add_event(
                        BallNetContactEvent(
                            timestamp=frame_number,
                            frame_number=frame_number,
                            contact_position=contact_position,
                            contact_height=contact_position[1],
                            impact_velocity=impact_velocity
                        )
                    )
                    
                    logger.info(f"Net contact detected at frame {frame_number}, position {contact_position}")
                    logger.info(f"   Direction change: {x_direction_change}, Velocity reduction: {velocity_reduction}, Upward bounce: {upward_bounce}")
                    return True
        
        return False
    
    def _detect_x_direction_change(self, vel1, vel2, vel3):
        """Detect if ball changed horizontal direction (bounced back from net)"""
        # Check if X velocity direction changed significantly
        if abs(vel1[0]) < 1 or abs(vel2[0]) < 1:  # Avoid division by zero
            return False
            
        # Check if direction reversed (positive to negative or vice versa)
        direction_reversed = (vel1[0] * vel3[0] < 0) and abs(vel3[0]) > abs(vel1[0]) * 0.3
        
        # Check if velocity significantly reduced then increased in opposite direction
        velocity_pattern = (abs(vel2[0]) < abs(vel1[0]) * 0.7) and (abs(vel3[0]) > abs(vel2[0]) * 1.2)
        
        return direction_reversed or velocity_pattern
    
    def _detect_velocity_reduction(self, vel1, vel2, vel3):
        """Detect if ball velocity was significantly reduced (indicating collision)"""
        speed1 = math.sqrt(vel1[0]**2 + vel1[1]**2)
        speed2 = math.sqrt(vel2[0]**2 + vel2[1]**2)
        speed3 = math.sqrt(vel3[0]**2 + vel3[1]**2)
        
        if speed1 < 2:  # Ignore very slow movements
            return False
        
        # Significant speed reduction followed by recovery
        speed_drop = speed2 < speed1 * 0.6  # Speed dropped by at least 40%
        speed_recovery = speed3 > speed2 * 1.3  # Speed increased by at least 30%
        
        return speed_drop and speed_recovery
    
    def _detect_upward_bounce(self, vel1, vel2, vel3):
        """Detect if ball bounced upward after hitting net"""
        # Ball was moving down/horizontally, then started moving up
        was_not_going_up = vel1[1] >= -2  # Not moving significantly upward
        now_going_up = vel3[1] < -5  # Now moving upward with significant speed
        
        # Check for trajectory curve typical of net bounce
        trajectory_curve = (vel1[1] >= vel2[1]) and (vel2[1] >= vel3[1]) and (vel3[1] < vel1[1] - 3)
        
        return (was_not_going_up and now_going_up) or trajectory_curve
    
    def _find_closest_point_to_net_center(self, trajectory_points, net_polygon):
        """Find the point in trajectory that is closest to the center of net polygon"""
        # Calculate net center
        net_center_x = sum(p[0] for p in net_polygon) / len(net_polygon)
        net_center_y = sum(p[1] for p in net_polygon) / len(net_polygon)
        net_center = (net_center_x, net_center_y)
        
        # Find closest trajectory point to net center
        closest_point = trajectory_points[0]
        min_distance = float('inf')
        
        for point in trajectory_points:
            distance = math.sqrt((point[0] - net_center[0])**2 + (point[1] - net_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                
        return closest_point
    
    def _line_intersects_polygon(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                                polygon: List[Tuple[float, float]]) -> bool:
        """Check if line segment p1-p2 intersects with any edge of the polygon"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        # Check intersection with each edge of the polygon
        n = len(polygon)
        for i in range(n):
            edge_start = polygon[i]
            edge_end = polygon[(i + 1) % n]
            
            if intersect(p1, p2, edge_start, edge_end):
                return True
        
        return False
