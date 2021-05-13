import cv2
from pose import Pose

# +++++++++++++===============================
WINDOW = "ML SHIT"
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
WINDOW_BRIGHTNESS = 150
VIDEO_CAPTURE_DEVICE = 0
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]
CENTER_BOX_HALF_SIZE = 128
POSE_CENTERED_SENSITIVITY = 50

class AI:
    def __init__(self):
        self.cmd = ''

    def update_cmd(self, cmd):
        if cmd != '':
            self.cmd = cmd

    @staticmethod
    def get_is_points_in_box(points, center_point):
        distance = {
            "x": 0,
            "y": 0
        }

        if center_point is not None and points is not None:
            for point in points:
                if point is not None:
                    if point[0] < center_point[0] - CENTER_BOX_HALF_SIZE:
                        distance['x'] = distance['x'] + (point[0] - (center_point[0] - CENTER_BOX_HALF_SIZE))
                    elif point[0] > center_point[0] + CENTER_BOX_HALF_SIZE:
                        distance['x'] = distance['x'] + (point[0] - (center_point[0] + CENTER_BOX_HALF_SIZE))

                    if point[1] < center_point[1] - CENTER_BOX_HALF_SIZE:
                        distance['y'] = distance['y'] + (point[1] - (center_point[1] - CENTER_BOX_HALF_SIZE))
                    elif point[1] > center_point[1] + CENTER_BOX_HALF_SIZE:
                        distance['y'] = distance['y'] + (point[1] - (center_point[1] + CENTER_BOX_HALF_SIZE))

        return distance


# ============== DRAWING

class Draw:
    def __init__(self):
        self.center_box_points = None
        self.center_point = None

    def get_center_box_points(self, frame):
        w = frame.shape[0]
        h = frame.shape[1]

        self.center_point = (h / 2, w / 2,)
        self.center_box_points = [
            (self.center_point[0] - CENTER_BOX_HALF_SIZE, self.center_point[1] - CENTER_BOX_HALF_SIZE),
            (self.center_point[0] + CENTER_BOX_HALF_SIZE, self.center_point[1] - CENTER_BOX_HALF_SIZE),
            (self.center_point[0] + CENTER_BOX_HALF_SIZE, self.center_point[1] + CENTER_BOX_HALF_SIZE),
            (self.center_point[0] - CENTER_BOX_HALF_SIZE, self.center_point[1] + CENTER_BOX_HALF_SIZE),
        ]

    def update_image(self, frame, cmd, draw_skeleton_flag, points, current_cmd, sum_of_absolute_distance):
        pose_line_color = (66, 144, 245)
        box_line_color = (0, 0, 245)

        # highlight skeleton when there is a cmd
        if current_cmd != '':
            pose_line_color = (185, 66, 245)

        # highlight box
        if POSE_CENTERED_SENSITIVITY > sum_of_absolute_distance['x'] > -POSE_CENTERED_SENSITIVITY and POSE_CENTERED_SENSITIVITY > \
                sum_of_absolute_distance['y'] > -POSE_CENTERED_SENSITIVITY:
            box_line_color = (185, 66, 245)

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
                            pose_line_color,
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
            self.center_point,
            8,
            box_line_color,
            thickness=-1,
            lineType=cv2.FILLED)

        # change line color if person is in box
        cv2.line(
            frame,
            self.center_box_points[0],
            self.center_box_points[1],
            box_line_color,
            2)
        cv2.line(
            frame,
            self.center_box_points[1],
            self.center_box_points[2],
            box_line_color,
            2)
        cv2.line(
            frame,
            self.center_box_points[2],
            self.center_box_points[3],
            box_line_color,
            2)
        cv2.line(
            frame,
            self.center_box_points[3],
            self.center_box_points[0],
            box_line_color,
            2)

        cv2.imshow(WINDOW, frame)


# ============== CAMERA
class Camera:
    def __init__(self):
        self.window_width = None
        self.window_height = None

        cv2.namedWindow(WINDOW)
        cv2.moveWindow(WINDOW, 600, 360) # do this dynaaimically
        self.feed = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        self.feed.set(10, WINDOW_BRIGHTNESS)
        self.feed.set(3, WINDOW_WIDTH)
        self.feed.set(4, WINDOW_HEIGHT)

    def set_window_size(self, frame):
        self.window_width = frame.shape[0]
        self.window_height = frame.shape[0]
        self.feed.set(3, self.window_width)
        self.feed.set(4, self.window_height)

    def start_feed(self, draw):
        while self.feed.isOpened():
            print('start feed loop')
            rval, frame = self.feed.read()

            if frame is not None:
                self.set_window_size(frame)
                draw.get_center_box_points(frame)
                break

    def process_feed(self, pose, ai, draw):
        while self.feed.isOpened():
            rval, frame = self.feed.read()

            if frame is not None:
                # each frame, run detection
                # each period, return the move likely pose
                cmd, draw_skeleton_flag, points = pose.detect(frame)
                sum_of_absolute_distance = ai.get_is_points_in_box(points, draw.center_point)
                ai.update_cmd(cmd)

                draw.update_image(frame, ai.cmd, draw_skeleton_flag, points, cmd, sum_of_absolute_distance)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Exit')
                break


# ============== MAIN
def main():
    ai = AI()
    pose = Pose()
    draw = Draw()
    camera = Camera()

    camera.start_feed(draw)
    camera.process_feed(pose, ai, draw)


if __name__ == "__main__":
    main()
