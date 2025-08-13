import cv2
from datatypes.track_object import BallTracker

class Annotator:
    def __init__(self, table_points, net_area):
        self.table_points = table_points
        self.net_area = net_area

    def annotate(self, image, ball_tracker: BallTracker, frame_number: int):
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

        # Left side points
        left_topLeft = self.table_points['left_side_points']['top_left']
        left_bottomLeft = self.table_points['left_side_points']['bottom_left']
        left_topRight = self.table_points['left_side_points']['top_right']
        left_bottomRight = self.table_points['left_side_points']['bottom_right']
        cv2.line(new_image, left_topLeft, left_bottomLeft, (255, 255, 0), 2)
        cv2.line(new_image, left_bottomLeft, left_bottomRight, (255, 255, 0), 2)
        cv2.line(new_image, left_bottomRight, left_topRight, (255, 255, 0), 2)
        cv2.line(new_image, left_topRight, left_topLeft, (255, 255, 0), 2)
        left_side_polygon = [left_topLeft, left_bottomLeft, left_bottomRight, left_topRight]

        # Right side points
        right_topLeft = self.table_points['right_side_points']['top_left']
        right_bottomLeft = self.table_points['right_side_points']['bottom_left']
        right_topRight = self.table_points['right_side_points']['top_right']
        right_bottomRight = self.table_points['right_side_points']['bottom_right']
        cv2.line(new_image, right_topLeft, right_bottomLeft, (255, 0, 255), 2)
        cv2.line(new_image, right_bottomLeft, right_bottomRight, (255, 0, 255), 2)
        cv2.line(new_image, right_bottomRight, right_topRight, (255, 0, 255), 2)
        cv2.line(new_image, right_topRight, right_topLeft, (255, 0, 255), 2)
        right_side_polygon = [right_topLeft, right_bottomLeft, right_bottomRight, right_topRight]

        # Draw net area if available
        net_top_left = self.net_area['top_left']
        net_top_right = self.net_area['top_right']
        net_bottom_left = self.net_area['bottom_left']
        net_bottom_right = self.net_area['bottom_right']
        
        # Draw net area in cyan color
        cv2.line(new_image, net_top_left, net_top_right, (255, 255, 0), 2)
        cv2.line(new_image, net_top_right, net_bottom_right, (255, 255, 0), 2)
        cv2.line(new_image, net_bottom_right, net_bottom_left, (255, 255, 0), 2)
        cv2.line(new_image, net_bottom_left, net_top_left, (255, 255, 0), 2)
        
        # Create net polygon for collision detection
        net_polygon = [net_top_left, net_bottom_left, net_bottom_right, net_top_right]

        new_image = ball_tracker.draw_tracking_info(new_image, left_side_polygon, right_side_polygon, frame_number=frame_number, net_polygon=net_polygon)

        return new_image