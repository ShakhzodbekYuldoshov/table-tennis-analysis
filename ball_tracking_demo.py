import cv2
import logging
import argparse
from config import config
from utils.annotator import Annotator
from datatypes.track_object import BallTracker
from datatypes.events import EventManager, EventType
from models.cv_technique import CVTechniqueBallDetector


logging.basicConfig(level=logging.INFO, filename='ball_tracking_demo.log',)
logger = logging.getLogger(__name__)


def on_table_contact(event):
    """Callback for table contact events"""
    print(f"Ball hit {event.table_side.value} side at {event.contact_position}")
    logger.info(f"Ball hit {event.table_side.value} side at {event.contact_position}")
    if event.bounce_height:
        print(f"   Bounce height: {event.bounce_height}")
        logger.info(f"   Bounce height: {event.bounce_height}")

def on_net_contact(event):
    """Callback for net contact events"""
    print(f"Ball hit the net at {event.contact_position}")
    logger.info(f"Ball hit the net at {event.contact_position}")

def on_out_of_bounds(event):
    """Callback for out of bounds events"""
    print(f"Ball went out of bounds at {event.exit_position}")
    logger.info(f"Ball went out of bounds at {event.exit_position}")


def main():
    parser = argparse.ArgumentParser(description='Ball Tracking Demo')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (optional)')
    
    args = parser.parse_args()

    # Initialize event manager
    event_manager = EventManager()
    # Register callbacks
    event_manager.register_callback(EventType.BALL_TABLE_CONTACT, on_table_contact)
    event_manager.register_callback(EventType.BALL_NET_CONTACT, on_net_contact)
    event_manager.register_callback(EventType.BALL_OUT_OF_BOUNDS, on_out_of_bounds)

    # Initialize detector with tracking
    detector = CVTechniqueBallDetector()
    ball_tracker = BallTracker(max_distance_threshold=100, min_tracking_frames=30, event_manager=event_manager)
    annotator = Annotator(config['table_points'], config['net_area'])
    
    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {args.source}")
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is provided
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # Detect and track objects
            detections = detector.detect(frame)
            ball_tracker.update(detections, frame_count)
            result_frame = annotator.annotate(frame, ball_tracker, frame_count)

            result_frame = cv2.resize(result_frame, (width//2, height//2))

            # Add frame information
            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(result_frame, info_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
            # Save or display frame
            if writer:
                writer.write(result_frame)
            else:
                cv2.imshow('Ball Tracking Demo', result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space to pause
                    cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"Processing completed!")
    print(f"Total frames processed: {frame_count}")
    
    # Final FPS statistics
    print(f"Processing completed!")
    print(f"Total frames processed: {frame_count}")
    
    if args.output:
        print(f"\nResult saved to: {args.output}")


if __name__ == "__main__":
    main()
