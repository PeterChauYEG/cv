CENTER_BOX_HALF_SIZE = 128
POSE_CENTERED_SENSITIVITY = 50

class AI:
    def __init__(self):
        self.cmd = ''
        self.is_pose_in_box = False

    def update_cmd(self, cmd):
        if cmd != '':
            self.cmd = cmd

    @staticmethod
    def get_sum_of_distance(points, center_point):
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

    def get_is_pose_in_box(self, sum_of_distance):
        if POSE_CENTERED_SENSITIVITY > sum_of_distance[
            'x'] > -POSE_CENTERED_SENSITIVITY and POSE_CENTERED_SENSITIVITY > \
                sum_of_distance['y'] > -POSE_CENTERED_SENSITIVITY:
            self.is_pose_in_box = True

        self.is_pose_in_box = False
