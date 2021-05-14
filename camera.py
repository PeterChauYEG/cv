import cv2

VIDEO_CAPTURE_DEVICE = 0
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
WINDOW_BRIGHTNESS = 150


class Camera:
    def __init__(self):
        self.feed = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        self.feed.set(3, WINDOW_WIDTH)
        self.feed.set(4, WINDOW_HEIGHT)
        self.feed.set(10, WINDOW_BRIGHTNESS)

    def get_frame(self):
        while self.feed.isOpened():
            rval, frame = self.feed.read()

            return frame
