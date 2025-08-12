import cv2
from datatypes.track_object import BallTracker

class Annotator:
    def __init__(self, table_points):
        self.table_points = table_points

    def annotate(self, image, ball_tracker: BallTracker):
        new_image = image.copy()
        top_left = self.table_points['top_left']
        top_right = self.table_points['top_right']
        bottom_left = self.table_points['bottom_left']
        bottom_right = self.table_points['bottom_right']

        # Draw the table outline
        cv2.line(new_image, top_left, top_right, (0, 255, 0), 2)
        cv2.line(new_image, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(new_image, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(new_image, bottom_left, top_left, (0, 255, 0), 2)

        new_image = ball_tracker.draw_tracking_info(new_image)

        return new_image