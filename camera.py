import cv2
from pose import Pose

#+++++++++++++===============================
WINDOW = "ML SHIT"
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
WINDOW_BRIGHTNESS = 150
VIDEO_CAPTURE_DEVICE = 1
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

class AI:
    def __init__(self):
        self.cmd = ''

    def update_cmd(self, cmd):
        if cmd != '':
            self.cmd = cmd

# ============== DRAWING
class Draw:
    def updateImage(self, frame, cmd, draw_skeleton_flag, points, current_cmd):
        line_color = (66, 144, 245)

        if frame is None or frame.size == 0:
            return

        # highlight skeleton when there is a cmd
        if current_cmd != '':
            line_color = (185, 66, 245)

        # Draw the detected skeleton points
        if draw_skeleton_flag == True:
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
            (WINDOW_WIDTH / 20, WINDOW_HEIGHT / 10),
            cv2.FONT_HERSHEY_PLAIN,
            1.75,
            (0, 0, 255),
            2)
        if cmd != '':
            cv2.putText(
                frame,
                cmd,
                (WINDOW_WIDTH / 20,( WINDOW_HEIGHT / 10) + 32),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (0, 0, 255),
                1)

        cv2.imshow(WINDOW, frame)

# ============== CAMERA
class Camera:
    def __init__(self):
        cv2.namedWindow(WINDOW)

        self.feed = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        self.feed.set(3, WINDOW_WIDTH)
        self.feed.set(4, WINDOW_HEIGHT)
        self.feed.set(10,WINDOW_BRIGHTNESS)

    def start(self, pose, ai, draw):
        while self.feed.isOpened():
            rval, frame = self.feed.read()

            while True:
                if frame is not None:
                    # each frame, run detection
                    # each period, return the move likely pose
                    cmd, draw_skeleton_flag, points = pose.detect(frame)
                    ai.update_cmd(cmd)

                    draw.updateImage(frame, ai.cmd, draw_skeleton_flag, points, cmd)

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
