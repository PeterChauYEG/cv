import cv2
import time
import math

'''
Correspondence between the number of the skeleton node and 
the human body:
Head - 0, Neck - 1, Right Shoulder - 2, Right Elbow - 3, Right Wrist - 4,
Left Shoulder - 5, Left Elbow - 6, Left Wrist - 7, Right Hip - 8,
Right Knee - 9, Right Ankle - 10, Left Hip - 11, Left Knee - 12,
Left Ankle - 13, Chest - 14, Background - 15
'''

# // -135 |\
# start = (50, 50)
# end = (100, 100)

# // -90 |
# start = (50, 50)
# end = (50, 100)

# // -45 /|
# start = (50, 50)
# end = (0, 100)

# // 0 -|
# start = (50, 50)
# end = (0, 50)

# // 45 \|
# start = (50, 50)
# end = (0, 0)

# // 90 ^
# start = (50, 50)
# end = (50, 0)

# // 135 |/
# start = (50, 50)
# end = (100, 0)

# // 180 |-
# start = (50, 50)
# end = (100, 50)
poses = {
    'arms_down_45': 'arms_down_45',
    'arms_flat': 'arms_flat',
    'arms_V': 'arms_V',
    'arms_up_45': 'arms_up_45',

    'left_arm_down_45': 'left_arm_down_45',  # 225 202.5,247.5
    'left_arm_flat': 'left_arm_flat',  # 180 202.5,157.5
    'left_arm_up_45': 'left_arm_up_45',  # 135 157.5,112.5

    'right_arm_down_45': 'right_arm_down_45',  # -45 -67.5,-22.5
    'right_arm_flat': 'right_arm_flat',  # 0 -22.5,22.5
    'right_arm_up_45': 'right_arm_up_45',  # 45 67.5,22.5

    # =======
    'left_arm_v': 'left_arm_v',  # 120 < shoulder_angle < 180
    'right_arm_v': 'right_arm_v',  # 20 < shoulder_angle < 80
}


class Pose:
    def __init__(self):

        # statics ============
        self.prob_threshold = 0.05

        # input image dimensions for the network
        # IMPORTANT:
        # Greater inWidth and inHeight will result in higher accuracy but longer process time
        # Smaller inWidth and inHeight will result in lower accuracy but shorter process time
        # Play around it by yourself to get best result!
        # https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

        # og
        # inWidth = 168
        # inHeight = 168
        self.input_w = 128
        self.input_h = 128

        # total number of the skeleton nodes
        self.nPoints = 15

        # read the path of the trained model of the neural network for pose recognition
        self.protoFile = "model/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = "model/pose/mpi/pose_iter_160000.caffemodel"

        # init vars ===========
        # input frame
        self.frame_w = None
        self.frame_h = None

        # count the number of frames
        self.frame_cnt = 0

        # posses detected
        self.poses_captured = {}

        # the period of pose reconigtion,it depends on your computer performance
        self.period = 0

        # record how many times the period of pose reconigtion called
        self.period_calculate_cnt = 0
        self.frame_cnt_threshold = 0
        self.pose_captured_threshold = 0

        # detection return
        self.draw_skeleton_flag = False
        self.cmd = ''
        self.points = []

        # init ===============
        # read the neural network of the pose recognition
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    @staticmethod
    def get_angle(start, end):
        """
        Calculate the angle between start and end

        :param start: start point [x, y]
        :param end: end point [x, y]
        :return: the clockwise angle from start to end
        """
        angle = int(math.atan2((start[1] - end[1]), (start[0] - end[0])) * 180 / math.pi)
        return angle

    def is_left_arm_up_45(self):
        left = False

        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 113 < shoulder_angle < 157:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True

        return left

    def is_right_arm_up_45(self):
        right = False

        if self.points[2] and self.points[3] and self.points[4]:
            shoulder_angle = self.get_angle(self.points[2], self.points[3])
            if 23 < shoulder_angle < 67:
                elbow_angle = self.get_angle(self.points[2], self.points[3])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        return right

    def is_left_arm_down_45(self):
        left = False

        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 203 < shoulder_angle < 247:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True

        return left

    def is_right_arm_down_45(self):
        right = False

        if self.points[2] and self.points[3] and self.points[4]:
            shoulder_angle = self.get_angle(self.points[2], self.points[3])
            if -67 < shoulder_angle < -23:
                elbow_angle = self.get_angle(self.points[2], self.points[3])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        return right

    def is_left_arm_flat(self):
        left = False

        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 158 < shoulder_angle < 202:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True

        return left

    def is_right_arm_flat(self):
        right = False

        if self.points[2] and self.points[3] and self.points[4]:
            shoulder_angle = self.get_angle(self.points[2], self.points[3])
            if -22 < shoulder_angle < 22:
                elbow_angle = self.get_angle(self.points[2], self.points[3])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        return right

    def is_arms_up_45(self):
        """
        Determine if the person is holding the arms
        like:
              \ | /
                |
               / \


        :return: if the person detected moves both of his arms up for about 45 degrees
        """
        right = False
        if self.points[2] and self.points[3] and self.points[4]:
            # calculate the shoulder angle
            shoulder_angle = self.get_angle(self.points[2], self.points[3])

            if 23 < shoulder_angle < 67:
                elbow_angle = self.get_angle(self.points[3], self.points[4])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        left = False
        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 113 < shoulder_angle < 157:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True

        if left and right:
            return True
        else:
            return False

    def is_arms_down_45(self):
        """
        Determine if the person is holding the arms
        like:
                |
              / | \
               / \


        :return: if the person detected moves both of his arms down for about 45 degrees
        """
        right = False
        if self.points[2] and self.points[3] and self.points[4]:
            # calculate the shoulder angle
            shoulder_angle = self.get_angle(self.points[2], self.points[3])

            if -67 < shoulder_angle < -23:
                elbow_angle = self.get_angle(self.points[3], self.points[4])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        left = False
        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 203 < shoulder_angle < 247:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True

        if left and right:
            return True
        else:
            return False

    def is_arms_flat(self):
        """
        Determine if the person moves his arm flat
        like: _ _|_ _
                 |
                / \
        :return: if the person detected moves both of his arms flat
        """
        right = False
        if self.points[2] and self.points[3] and self.points[4]:

            shoulder_angle = self.get_angle(self.points[2], self.points[3])
            # if arm is flat
            if -22 < shoulder_angle < 22:
                elbow_angle = self.get_angle(self.points[3], self.points[4])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 30:
                    right = True

        left = False
        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the  dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            # if arm is flat
            if 158 < shoulder_angle < 202:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 30:
                    left = True

        if left and right:
            return True
        else:
            return False

    def is_arms_v(self):
        """
        Determine if the person has his/her shoulder and elbow to a certain degree
        like:   |
              \/|\/
               / \

        """
        right = False

        if self.points[2] and self.points[3] and self.points[4]:
            shoulder_angle = self.get_angle(self.points[2], self.points[3])

            if -67 < shoulder_angle < -23:
                elbow_angle = self.get_angle(self.points[3], self.points[4])
                if 0 < elbow_angle < 90:
                    right = True

        left = False
        if self.points[5] and self.points[6] and self.points[7]:
            shoulder_angle = self.get_angle(self.points[5], self.points[6])
            # correct the  dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360
            if 203 < shoulder_angle < 247:
                elbow_angle = self.get_angle(self.points[6], self.points[7])
                if 90 < elbow_angle < 180:
                    left = True

        if left and right:
            return True
        else:
            return False

    def preprocess(self, frame):
        frame = cv2.bilateralFilter(frame, 5, 50, 100)

        if self.frame_w is None or self.frame_h is None:
            self.frame_w = frame.shape[1]
            self.frame_h = frame.shape[0]

        frame_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.input_w, self.input_h),
                                           (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(frame_blob)

        return frame

    def calculate_period(self, period_start_time, period_end_time):
        if self.period_calculate_cnt <= 5:
            self.period = self.period + period_end_time - period_start_time
            self.period_calculate_cnt = self.period_calculate_cnt + 1
        if self.period_calculate_cnt >= 6:
            self.period = self.period / 6

    def set_detection_thresholds(self):
        if self.period < 0.3:
            self.frame_cnt_threshold = 5
            self.pose_captured_threshold = 4
        elif 0.3 <= self.period < 0.6:
            self.frame_cnt_threshold = 4
            self.pose_captured_threshold = 3
        elif self.period >= 0.6:
            self.frame_cnt_threshold = 2
            self.pose_captured_threshold = 2

    def clear_detection_period_state(self):
        self.frame_cnt = 0
        self.poses_captured = {}

    def clear_detection_state(self):
        self.draw_skeleton_flag = False
        self.cmd = ''
        self.points = []

    def calculate_period_cmd(self):
        if self.frame_cnt >= self.frame_cnt_threshold:
            if len(self.poses_captured) != 0:
                pose = max(self.poses_captured, key=lambda k: self.poses_captured[k])

                # we need a map of pose to cmd
                if pose != '':
                    print(pose)
                    self.cmd = pose

            self.clear_detection_period_state()

    def update_poses_captured(self, key):
        if not self.poses_captured.has_key(key):
            self.poses_captured[key] = 0
        self.poses_captured[key] += 1
        print "%d:%s captured" % (self.frame_cnt, key)

    def calculate_pose(self):
        if self.is_arms_down_45():
            self.update_poses_captured(poses['arms_down_45'])
        elif self.is_arms_flat():
            self.update_poses_captured(poses['arms_flat'])
        elif self.is_arms_v():
            self.update_poses_captured(poses['arms_V'])
        elif self.is_arms_up_45():
            self.update_poses_captured(poses['arms_up_45'])
        elif self.is_left_arm_up_45():
            self.update_poses_captured(poses['left_arm_up_45'])
        elif self.is_right_arm_up_45():
            self.update_poses_captured(poses['right_arm_up_45'])
        elif self.is_left_arm_down_45():
            self.update_poses_captured(poses['left_arm_down_45'])
        elif self.is_right_arm_down_45():
            self.update_poses_captured(poses['right_arm_down_45'])
        elif self.is_left_arm_flat():
            self.update_poses_captured(poses['left_arm_flat'])
        elif self.is_right_arm_flat():
            self.update_poses_captured(poses['right_arm_flat'])

    def handle_pose_points(self, output):
        # get shape of the output
        H = output.shape[2]
        W = output.shape[3]

        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (self.frame_w * point[0]) / W
            y = (self.frame_h * point[1]) / H

            if prob > self.prob_threshold:
                self.points.append((int(x), int(y)))
                self.draw_skeleton_flag = True
            else:
                self.points.append(None)
                self.draw_skeleton_flag = False

    def detect(self, frame):
        """
        Main operation to recognize body pose using a trained model

        :param frame: raw h264 decoded frame
        :return:
                draw_skeleton_flag: the flag that indicates if the skeleton
                are detected and depend if the skeleton is drawn on the pic
                cmd: the command to be received by Tello
                points:the coordinates of the skeleton nodes
        """
        period_start_time = 0
        self.clear_detection_state()

        self.preprocess(frame)

        # get the output of the neural network and calculate the period of the process
        if self.period is 0:
            period_start_time = time.time()

        output = self.net.forward()

        if self.period is 0:
            period_end_time = time.time()

            # calculation the period of pose reconigtion for 6 times,and get the average value
            self.calculate_period(period_start_time, period_end_time)

            # set the frame_cnt_threshold and pose_captured_threshold according to
            # the period of the pose recognition
            self.set_detection_thresholds()

        # calculate the detected points
        self.handle_pose_points(output)

        # check the captured pose
        self.calculate_pose()

        # inc frame counter
        self.frame_cnt += 1

        # check whether pose control command are generated once for
        # certain times of pose recognition
        self.calculate_period_cmd()

        return self.cmd, self.draw_skeleton_flag, self.points
