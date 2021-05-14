CENTER_BOX_HALF_SIZE = 128
POSE_CENTERED_SENSITIVITY = 50


class AI:
    def __init__(self):
        self.current_pose = ''
        self.is_pose_in_box = False
        self.distance = {
            "x": 0,
            "y": 0
        }
        self.drone_cmd = ''

    def reset_state(self):
        self.current_pose = ''
        self.is_pose_in_box = False
        self.distance = {
            "x": 0,
            "y": 0
        }
        self.drone_cmd = ''

    def calculate_drone_cmd(self):
        if self.is_pose_in_box is False:
            if self.distance['x'] < 0:
                self.drone_cmd = 'x-dec'
            if self.distance['x'] > 0:
                self.drone_cmd = 'x-inc'
            if self.distance['y'] < 0:
                self.drone_cmd = 'y-inc'
            if self.distance['y'] > 0:
                self.drone_cmd = 'y-dec'

    def update_current_pose(self, current_pose):
        self.current_pose = current_pose

    def get_sum_of_distance(self, points, center_point):
        if center_point is not None and points is not None:
            for point in points:
                if point is not None:
                    if point[0] < center_point[0] - CENTER_BOX_HALF_SIZE:
                        self.distance['x'] = self.distance['x'] + (point[0] - (center_point[0] - CENTER_BOX_HALF_SIZE))
                    elif point[0] > center_point[0] + CENTER_BOX_HALF_SIZE:
                        self.distance['x'] = self.distance['x'] + (point[0] - (center_point[0] + CENTER_BOX_HALF_SIZE))

                    if point[1] < center_point[1] - CENTER_BOX_HALF_SIZE:
                        self.distance['y'] = self.distance['y'] + (point[1] - (center_point[1] - CENTER_BOX_HALF_SIZE))
                    elif point[1] > center_point[1] + CENTER_BOX_HALF_SIZE:
                        self.distance['y'] = self.distance['y'] + (point[1] - (center_point[1] + CENTER_BOX_HALF_SIZE))

    def get_is_pose_in_box(self):
        if POSE_CENTERED_SENSITIVITY > self.distance[
            'x'] > -POSE_CENTERED_SENSITIVITY and POSE_CENTERED_SENSITIVITY > \
                self.distance['y'] > -POSE_CENTERED_SENSITIVITY:
            self.is_pose_in_box = True

        self.is_pose_in_box = False
