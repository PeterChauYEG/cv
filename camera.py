import cv2
from pose import Pose

# +++++++++++++===============================
WINDOW = "ML SHIT"
WINDOW_WIDTH = 1280 # write code to get and set frame w and h once on start
WINDOW_HEIGHT = 720
WINDOW_BRIGHTNESS = 150
VIDEO_CAPTURE_DEVICE = 0
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]
center_box_half_size = 128


class AI:
    def __init__(self):
        self.cmd = ''

    def update_cmd(self, cmd):
        if cmd != '':
            self.cmd = cmd


# ============== DRAWING

class Draw:
    def __init__(self):
        pass

    @staticmethod
    def get_center_box_points(frame):
        w = frame.shape[0]
        h = frame.shape[1]

        center_point = (h / 2, w / 2,)
        center_box_points = [
            (center_point[0] - center_box_half_size, center_point[1] - center_box_half_size),
            (center_point[0] + center_box_half_size, center_point[1] - center_box_half_size),
            (center_point[0] + center_box_half_size, center_point[1] + center_box_half_size),
            (center_point[0] - center_box_half_size, center_point[1] + center_box_half_size),
        ]

        return center_point, center_box_points

    @staticmethod
    def resize_frame(frame):
        if frame is None or frame.size == 0:
            return

        scale_percent = 60
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        return resized_frame

    def update_image(self, frame, cmd, draw_skeleton_flag, points, current_cmd):
        line_color = (66, 144, 245)

        if frame is None or frame.size == 0:
            return

        center_point, center_box_points = self.get_center_box_points(frame)

        # highlight skeleton when there is a cmd
        if current_cmd != '':
            line_color = (185, 66, 245)

        # Draw the detected skeleton points
        if draw_skeleton_flag:
            for i in range(15):
                # Draw Skeleton
                for pair in POSE_PAIRS:
                    partA = pair[0]
                    partB = pair[1]
                    if points[partA] and points[partB]:
                        cv2.line(
                            frame,
                            points[partA],
                            points[partB],
                            line_color,
                            2)

                # Draw points
                cv2.circle(
                    frame,
                    points[i],
                    8,
                    (185, 66, 245),
                    thickness=-1,
                    lineType=cv2.FILLED)

                # Draw Labels
                cv2.putText(
                    frame,
                    "{}".format(i),
                    points[i],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (245, 66, 227),
                    1,
                    lineType=cv2.LINE_AA)

        # DRAW COMMAND
        cv2.putText(
            frame,
            'Drone AI',
            (16, 16),
            cv2.FONT_HERSHEY_PLAIN,
            1.75,
            (0, 0, 255),
            2)
        if cmd != '':
            cv2.putText(
                frame,
                cmd,
                (16, 48),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (0, 0, 255),
                1)

        # DRAW Center
        cv2.circle(
            frame,
            center_point,
            8,
            (255, 128, 245),
            thickness=-1,
            lineType=cv2.FILLED)

# change line color if person is in box
        cv2.line(
            frame,
            center_box_points[0],
            center_box_points[1],
            (255, 128, 245),
            2)
        cv2.line(
            frame,
            center_box_points[1],
            center_box_points[2],
            (255, 128, 245),
            2)
        cv2.line(
            frame,
            center_box_points[2],
            center_box_points[3],
            (255, 128, 245),
            2)
        cv2.line(
            frame,
            center_box_points[3],
            center_box_points[0],
            (255, 128, 245),
            2)

        cv2.imshow(WINDOW, frame)


# ============== CAMERA
class Camera:
    def __init__(self):
        cv2.namedWindow(WINDOW)

        self.feed = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        self.feed.set(3, WINDOW_WIDTH)
        self.feed.set(4, WINDOW_HEIGHT)
        self.feed.set(10, WINDOW_BRIGHTNESS)

    def start(self, pose, ai, draw):
        while self.feed.isOpened():
            rval, frame = self.feed.read()

            while True:
                if frame is not None:
                    # each frame, run detection
                    # each period, return the move likely pose
                    cmd, draw_skeleton_flag, points = pose.detect(frame)
                    ai.update_cmd(cmd)

                    # resize_frame = draw.resize_frame(frame)
                    draw.update_image(frame, ai.cmd, draw_skeleton_flag, points, cmd)

                rval, frame = self.feed.read()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# ============== MAIN
def main():
    ai = AI()
    pose = Pose()
    draw = Draw()
    camera = Camera()

    camera.start(pose, ai, draw)


if __name__ == "__main__":
    main()
