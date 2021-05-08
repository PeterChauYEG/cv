import cv2
import time
import math
import numpy as np
import datetime
import os

'''
Correspondence between the number of the skeleton node and 
the human body:
Head - 0, Neck - 1, Right Shoulder - 2, Right Elbow - 3, Right Wrist - 4,
Left Shoulder - 5, Left Elbow - 6, Left Wrist - 7, Right Hip - 8,
Right Knee - 9, Right Ankle - 10, Left Hip - 11, Left Knee - 12,
Left Ankle - 13, Chest - 14, Background - 15
'''

class Pose:
    def __init__(self):

        # read the path of the trained model of the neural network for pose recognition
        self.protoFile = "model/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = "model/pose/mpi/pose_iter_160000.caffemodel"

        # read the neural network of the pose recognition
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

        # total number of the skeleton nodes
        self.nPoints = 15

        # init vars
        self.frame_w = None
        self.frame_h = None

        # count the number of frames,and after every certain number of frames
        # is read, frame_cnt will be cleared and recounted.
        self.frame_cnt = 0
        self.arm_down_45_cnt = 0 # count numbers of the arm_dowm_45 captured in every certain number of frames
        self.arm_flat_cnt = 0    # count numbers of the arm_flat captured in every certain number of frames
        self.arm_V_cnt = 0       # count numbers of the arm_V captured in every certain number of frames

        # the period of pose reconigtion,it depends on your computer performance
        self.period = 0

        # record how many times the period of pose reconigtion is calculated.
        self.period_calculate_cnt = 0
        self.frame_cnt_threshold = 0
        self.pose_captured_threshold = 0

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

        self.prob_threshold = 0.05

        # detection returrn
        self.draw_skeleton_flag = False
        self.cmd = ''
        self.points = []

    def getAngle(self, start, end):
        """
        Calculate the angle between start and end

        :param start: start point [x, y]
        :param end: end point [x, y]
        :return: the clockwise angle from start to end
        """
        angle = int(math.atan2((start[1] - end[1]), (start[0] - end[0])) * 180 / math.pi)
        return angle

    def is_arms_down_45(self, points):
        """
        Determine if the person is holding  the arms
        like:
                |
              / | \
               / \


        :param points: set of body key points
        :return: if the person detected moves both of his arms down for about 45 degrees
        """
        right = False
        if points[2] and points[3] and points[4]:
            # calculate the shoulder angle
            shoulder_angle = self.getAngle(points[2], points[3])

            if -60 < shoulder_angle < -20:
                elbow_angle = self.getAngle(points[3], points[4])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    right = True

        left = False
        if points[5] and points[6] and points[7]:
            shoulder_angle = self.getAngle(points[5], points[6])
            # correct the dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            if 200 < shoulder_angle < 240:
                elbow_angle = self.getAngle(points[6], points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 25:
                    left = True
        # If at least one arm meets the requirements, it is considered a successful capture
        if left or right:
            return True
        else:
            return False

    def is_arms_flat(self, points):
        """
        Determine if the person moves his arm flat
        like: _ _|_ _
                 |
                / \
        :param points: set of body key points
        :return: if the person detected moves both of his arms flat
        """
        right = False
        if points[2] and points[3] and points[4]:

            shoulder_angle = self.getAngle(points[2], points[3])
            # if arm is flat
            if -10 < shoulder_angle < 40:
                elbow_angle = self.getAngle(points[3], points[4])
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 30:
                    right = True

        left = False
        if points[5] and points[6] and points[7]:
            shoulder_angle = self.getAngle(points[5], points[6])
            # correct the  dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360

            # if arm is flat
            if 140 < shoulder_angle < 190:
                elbow_angle = self.getAngle(points[6], points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                # if arm is straight
                if abs(elbow_angle - shoulder_angle) < 30:
                    left = True
        # If at least one arm meets the requirements, it is considered a successful capture
        if left or right:
            return True
        else:
            return False

    def is_arms_V(self, points):
        """
        Determine if the person has his/her shoulder and elbow to a certain degree
        like:   |
              \/|\/
               / \

        :param points: set of body key pointss
        """
        right = False

        if points[2] and points[3] and points[4]:
            shoulder_angle = self.getAngle(points[2], points[3])

            if -60 < shoulder_angle < -20:
                elbow_angle = self.getAngle(points[3], points[4])
                if 0 < elbow_angle < 90 :
                    right = True

        left = False
        if points[5] and points[6] and points[7]:
            shoulder_angle = self.getAngle(points[5], points[6])
            # correct the  dimension
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360
            if 200 < shoulder_angle < 240:
                elbow_angle = self.getAngle(points[6], points[7])
                if  90 < elbow_angle < 180:
                    left = True

        # If at least one arm meets the requirements, it is considered a successful capture
        if left or right:
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

    def calculate_period(self, period_starttime, period_endtime):
        if self.period_calculate_cnt <= 5 :
            self.period = self.period + period_endtime - period_starttime
            self.period_calculate_cnt = self.period_calculate_cnt + 1
        if self.period_calculate_cnt >= 6 :
            self.period = self.period / 6

    def set_detection_thresholds(self):
        if self.period < 0.3:
            self.frame_cnt_threshold = 5
            self.pose_captured_threshold = 4
        elif self.period >= 0.3 and self.period <0.6:
            self.frame_cnt_threshold = 4
            self.pose_captured_threshold = 3
        elif self.period >= 0.6:
            self.frame_cnt_threshold = 2
            self.pose_captured_threshold = 2

    def clear_detection_period_state(self):
        self.frame_cnt = 0
        self.arm_down_45_cnt = 0
        self.arm_flat_cnt = 0
        self.arm_V_cnt = 0

    def clear_detection_state(self):
        self.draw_skeleton_flag = False
        self.cmd = ''
        self.points = []

    def calculate_period_cmd(self):
        if self.frame_cnt >= self.frame_cnt_threshold:
            if self.arm_down_45_cnt >= self.pose_captured_threshold:
                print '!!!arm up,move back!!!'
                self.cmd =  'moveback'
            elif self.arm_flat_cnt >= self.pose_captured_threshold:
                print '!!!arm down,moveforward!!!'
                self.cmd =  'moveforward'
            elif self.arm_V_cnt >= self.pose_captured_threshold :
                print '!!!arm V,land!!!'
                self.cmd =  'land'
            self.clear_detection_period_state()

    def calculate_pose(self):
        if self.is_arms_down_45(self.points):
            self.arm_down_45_cnt += 1
            print "%d:arm down captured"%self.frame_cnt

        if self.is_arms_flat(self.points):
            self.arm_flat_cnt += 1
            print "%d:arm up captured"%self.frame_cnt

        if self.is_arms_V(self.points):
            self.arm_V_cnt += 1
            print '%d:arm V captured'%self.frame_cnt

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
        period_starttime = 0
        self.clear_detection_state()

        self.preprocess(frame)

        # get the output of the neural network and calculate the period of the process
        if self.period is 0:
            period_starttime = time.time()

        output = self.net.forward()

        if self.period is 0:
            period_endtime = time.time()

             # calculation the period of pose reconigtion for 6 times,and get the average value
            self.calculate_period(period_starttime, period_endtime)

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
