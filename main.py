import cv2

from camera import Camera
from pose import Pose
from GUI import GUI
from ai import AI


def process_feed(pose, ai, gui, frame):
    # each frame, run detection
    # each period, return the move likely pose
    current_pose, gui_skeleton_flag, points = pose.detect(frame)
    ai.update_current_pose(current_pose)

    ai.get_sum_of_distance(points, gui.center_point)
    ai.get_is_pose_in_box()

    ai.calculate_drone_cmd()

    gui.update_image(frame, gui_skeleton_flag, points, ai)

    ai.reset_state()

def main():
    ai = AI()
    pose = Pose()
    camera = Camera()
    gui = GUI()

    # get first frame
    frame = camera.get_frame()

    # initial setup
    if frame is not None:
        gui.set_window_size(frame, camera)
        gui.get_center_box_points(frame)

    # main loop
    while camera.feed.isOpened():
        if frame is not None:
            process_feed(pose, ai, gui, frame)

        # get next frame
        frame = camera.get_frame()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exit')
            break


if __name__ == "__main__":
    main()
