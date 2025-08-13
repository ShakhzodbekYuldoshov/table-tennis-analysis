"""
Event classes for table tennis analysis.
These classes represent different events that occur during gameplay.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque
from enum import Enum
import time


class TableSide(Enum):
    """Enum for table sides"""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Enum for different event types"""
    BALL_TABLE_CONTACT = "ball_table_contact"
    BALL_NET_CONTACT = "ball_net_contact"
    BALL_OUT_OF_BOUNDS = "ball_out_of_bounds"
    BALL_TRAJECTORY = "ball_trajectory"


@dataclass
class BaseEvent:
    """Base class for all table tennis events"""
    timestamp: float  # Time when event occurred
    frame_number: int  # Frame number when event was detected
    event_type: EventType = field(init=False)  # Will be set in subclasses
    confidence: float = 1.0  # Confidence level of event detection (0-1)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class BallTableContactEvent(BaseEvent):
    """Event for when ball touches the table"""
    contact_position: Tuple[float, float] = None  # (x, y) coordinates of contact
    table_side: TableSide = None  # Which side of the table was contacted
    bounce_height: Optional[float] = None  # Height of bounce if detectable
    impact_velocity: Optional[Tuple[float, float]] = None  # (vx, vy) velocity at impact
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.BALL_TABLE_CONTACT


@dataclass
class BallNetContactEvent(BaseEvent):
    """Event for when ball touches the net"""
    contact_position: Tuple[float, float] = None  # (x, y) coordinates of contact
    contact_height: Optional[float] = None  # Height at which ball hit net
    impact_velocity: Optional[Tuple[float, float]] = None  # (vx, vy) velocity at impact
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.BALL_NET_CONTACT


@dataclass
class BallOutOfBoundsEvent(BaseEvent):
    """Event for when ball goes out of bounds"""
    exit_position: Tuple[float, float] = None  # (x, y) coordinates where ball left table area
    exit_velocity: Optional[Tuple[float, float]] = None  # (vx, vy) velocity when leaving
    last_table_contact: Optional['BallTableContactEvent'] = None  # Reference to last table contact
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.BALL_OUT_OF_BOUNDS


@dataclass
class BallTrajectoryPoint:
    """Single point in ball trajectory"""
    position: Tuple[float, float]  # (x, y) coordinates
    timestamp: float  # Time of this position
    frame_number: int  # Frame number
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) velocity at this point
    acceleration: Optional[Tuple[float, float]] = None  # (ax, ay) acceleration at this point


@dataclass
class BallTrajectoryEvent(BaseEvent):
    """Event for tracking ball trajectory over time"""
    trajectory_points: List[BallTrajectoryPoint] = field(default_factory=list)  # List of trajectory points
    start_position: Tuple[float, float] = (0.0, 0.0)  # Starting position
    end_position: Tuple[float, float] = (0.0, 0.0)  # Ending position
    duration: float = 0.0  # Duration of trajectory in seconds
    average_velocity: Optional[Tuple[float, float]] = None  # Average velocity over trajectory
    max_height: Optional[float] = None  # Maximum height reached during trajectory
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.BALL_TRAJECTORY
        
        if self.trajectory_points:
            self.start_position = self.trajectory_points[0].position
            self.end_position = self.trajectory_points[-1].position
            self.duration = self.trajectory_points[-1].timestamp - self.trajectory_points[0].timestamp
    
    def add_point(self, point: BallTrajectoryPoint):
        """Add a new point to the trajectory"""
        self.trajectory_points.append(point)
        self.end_position = point.position
        if self.trajectory_points:
            self.duration = point.timestamp - self.trajectory_points[0].timestamp
    
    def calculate_average_velocity(self) -> Optional[Tuple[float, float]]:
        """Calculate average velocity over the trajectory"""
        if len(self.trajectory_points) < 2:
            return None
        
        start_point = self.trajectory_points[0]
        end_point = self.trajectory_points[-1]
        
        if self.duration <= 0:
            return None
        
        dx = end_point.position[0] - start_point.position[0]
        dy = end_point.position[1] - start_point.position[1]
        
        avg_vx = dx / self.duration
        avg_vy = dy / self.duration
        
        self.average_velocity = (avg_vx, avg_vy)
        return self.average_velocity
    
    def find_max_height(self) -> Optional[float]:
        """Find maximum height (minimum y-value) in trajectory"""
        if not self.trajectory_points:
            return None
        
        min_y = min(point.position[1] for point in self.trajectory_points)
        self.max_height = min_y
        return self.max_height


class EventManager:
    """Manager class for handling table tennis events"""
    
    def __init__(self):
        self.events: List[BaseEvent] = deque(maxlen=10)
        self.event_callbacks = {}
    
    def add_event(self, event: BaseEvent):
        """Add a new event to the manager"""
        self.events.append(event)
        
        # Trigger callbacks if registered
        if event.event_type in self.event_callbacks:
            for callback in self.event_callbacks[event.event_type]:
                callback(event)
    
    def register_callback(self, event_type: EventType, callback):
        """Register a callback function for specific event type"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def get_events(self) -> List[BaseEvent]:
        """Get all events"""
        return self.events
    
    def get_events_by_type(self, event_type: EventType) -> List[BaseEvent]:
        """Get all events of specific type"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_in_time_range(self, start_time: float, end_time: float) -> List[BaseEvent]:
        """Get all events within specified time range"""
        return [event for event in self.events 
                if start_time <= event.timestamp <= end_time]
    
    def get_table_contacts_by_side(self, side: TableSide) -> List[BallTableContactEvent]:
        """Get all table contact events for specific side"""
        table_events = self.get_events_by_type(EventType.BALL_TABLE_CONTACT)
        return [event for event in table_events if event.table_side == side]
    
    def clear_events(self):
        """Clear all events"""
        self.events.clear()
    
    def get_latest_event(self, event_type: EventType) -> Optional[BaseEvent]:
        """Get the most recent event of specified type"""
        events = self.get_events_by_type(event_type)
        if events:
            return max(events, key=lambda e: e.timestamp)
        return None
    
    def get_event_count(self, event_type: EventType) -> int:
        """Get count of events of specific type"""
        return len(self.get_events_by_type(event_type))
    
    def export_events_summary(self) -> dict:
        """Export a summary of all events"""
        summary = {
            "total_events": len(self.events),
            "events_by_type": {},
            "table_contacts_by_side": {
                TableSide.LEFT.value: 0,
                TableSide.RIGHT.value: 0,
                TableSide.UNKNOWN.value: 0
            }
        }
        
        # Count events by type
        for event_type in EventType:
            summary["events_by_type"][event_type.value] = self.get_event_count(event_type)
        
        # Count table contacts by side
        for side in TableSide:
            summary["table_contacts_by_side"][side.value] = len(self.get_table_contacts_by_side(side))
        
        return summary
