import cv2
from ultralytics import YOLO


class CVTechniqueBallDetector:
    def __init__(self):
        self.model = YOLO("yolo11s.pt")
        self.prev_image = None
    
    def detect(self, image):
        # Placeholder for detection logic using OpenCV techniques
        filtered_contours = []

        # detect person to filter person's moving places
        person_detections = self.model(image, classes=[0], verbose=False)[0]   # detect only person
        person_boxes = person_detections.boxes.xyxy.cpu().numpy() if person_detections.boxes else []

        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        if self.prev_image is not None:
            # Subtract current image from previous one to detect changes
            diff = cv2.subtract(gray, self.prev_image)

            # draw black rectangles around detected persons
            for person in person_boxes:
                x1, y1, x2, y2 = person.tolist()
                cv2.rectangle(diff, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

            # process image to find contours
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            structuringElementSize = (7, 7)
            structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, structuringElementSize)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, structuringElement)
            finalThresholdImage = cv2.GaussianBlur(thresh, (5, 5), cv2.BORDER_DEFAULT)
            perimeterMin = 50
            perimeterMax = 135

            # instead of getting a tree of contours (ie, each contour contain a child)
            # contours, hierarchy = cv2.findContours(finalThresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # we can get only top levels contours
            contours, hierarchy = cv2.findContours(finalThresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Select contours with specific arc length
            real_cnts = []

            for cnt in contours:
                perimeter = cv2.arcLength(cnt, True)
                if perimeterMin < perimeter < perimeterMax:
                    real_cnts.append(cnt)

            for c in real_cnts:
                segmentation_area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                # if segmentation takes more than 90% of the bounding box area, consider it a valid detection
                if segmentation_area < 0.6 * w * h:
                    continue

                filtered_contours.append(c)

        self.prev_image = gray
        return filtered_contours
