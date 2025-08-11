import cv2
import time
import logging
from config import table_points

def main(
        video: str = "video.mp4",
):
    fps = 0
    frame_counter = 0
    # Load a video file
    cap = cv2.VideoCapture(video)

    # Check if the video was opened successfully
    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        return

    # Read and display the video frames
    while True:
        main_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        main_end_time = time.time()

        # draw table points on the frame
        for point in table_points.values():
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
        
        # draw lines between the points
        cv2.line(frame, table_points["top_left"], table_points["top_right"], (255, 0, 0), 2)
        cv2.line(frame, table_points["top_right"], table_points["bottom_right"], (255, 0, 0), 2)
        cv2.line(frame, table_points["bottom_right"], table_points["bottom_left"], (255, 0, 0), 2)
        cv2.line(frame, table_points["bottom_left"], table_points["top_left"], (255, 0, 0), 2)

        # Calculate and display FPS
        fps = 1 / (main_end_time - main_start_time)
        logging.info(f"FPS: {fps:.2f}")
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))  # Resize frame to 1024x768
        cv2.imshow("Video", frame)
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.info("Starting video processing...")
    main(
        video="videos/IMG_7370.MOV",
    )
    logging.info("Video processing completed.")
