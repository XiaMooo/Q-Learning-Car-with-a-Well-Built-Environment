from utils import *

k = 5
f = -0.15
miu = -0.005
dt = 0.5

PATH_WIDTH = 250


class Environment:
    def __init__(self,
                 tot_length=5000,
                 path_width=PATH_WIDTH,
                 length_range=(600, 900),
                 dir_range=(np.pi / 6, np.pi / 3),
                 ):

        self.tot_length = tot_length
        self.path_width = path_width
        self.length_range = length_range
        self.dir_range = dir_range

        self.path_points, self.path_directions = self.generate_path(self.tot_length,
                                                                    self.length_range,
                                                                    self.dir_range
                                                                    )

        self.road_polygon = self.create_road(self.path_directions, self.path_points, self.path_width)

    def reset(self, car, recreate_road=False):
        if recreate_road:
            self.path_points, self.path_directions = self.generate_path(self.tot_length,
                                                                        self.length_range,
                                                                        self.dir_range
                                                                        )
            self.road_polygon = self.create_road(self.path_directions, self.path_points, self.path_width)

        car.clear_hist()
        car.set_pos_dir_v(self.path_points[1] / 10,
                          theta_e(self.path_directions[0, 1] + np.random.uniform(-0.1, 0.1)),
                          0.0)

    def is_in_road(self, p):
        for i in range(len(self.path_directions) - 1, -1, -1):
            quadrangle = self.road_polygon[[i, i + 1, -3 - i, -2 - i]]
            in_path = in_convex_polygon(p, quadrangle)

            if in_path:
                return i

        return -1

    def completion(self, p):
        ind = self.is_in_road(p)
        if ind == -1:
            return ind, 0, 0
        passed = np.sum(self.path_directions[:ind, 0])
        line = self.path_points[[ind, ind + 1]]
        length = pos_normal(p, line)

        return ind, passed + length, length

    def observe(self, car) -> tuple[bool, np.ndarray]:
        point = car.position
        direction = car.direction
        radar_dirs = car.radar_directions

        ind = self.is_in_road(car.position)
        if ind == -1:
            return False, np.zeros((len(radar_dirs), 3))

        radar_pts = np.vstack([point] * len(radar_dirs))
        r_theta = np.ones((len(radar_dirs), 2)) * self.path_width * 2
        r_theta[..., 1] = np.array(radar_dirs) + dir2theta(direction)
        radar_pts += r_theta_2_x_y(r_theta)

        result = np.zeros((len(radar_dirs), 3))
        result[..., :2] = radar_pts.copy()
        result[..., 2] = self.path_width * 2

        if ind == 0:
            pts_l = [0, 1, 2]
            pts_r = [-4, -3, -2, -1]
        elif ind == len(self.road_polygon) // 2 - 2:
            pts_l = [ind - 1, ind, ind + 1, ind + 2]
            pts_r = [ind + 2, ind + 3, ind + 4]
        else:
            pts_l = [ind - 1, ind, ind + 1, ind + 2]
            pts_r = [-1 - ind, -2 - ind, -3 - ind, -4 - ind]

        for i, radar_pt in enumerate(radar_pts):
            for j in range(len(pts_l) - 1):
                q1 = self.road_polygon[pts_l[j]]
                q2 = self.road_polygon[pts_l[j + 1]]
                cb, cp = point_cross(point, radar_pt, q1, q2)
                if cb and (r := mod(cp - point)) < result[i, 2]:
                    result[i, :2] = cp
                    result[i, 2] = r
            for j in range(len(pts_r) - 1):
                q1 = self.road_polygon[pts_r[j]]
                q2 = self.road_polygon[pts_r[j + 1]]
                cb, cp = point_cross(point, radar_pt, q1, q2)
                if cb and (r := mod(cp - point)) < result[i, 2]:
                    result[i, :2] = cp
                    result[i, 2] = r

        return True, result

    @staticmethod
    def generate_path(tot_length, length_range, dir_range):
        path_pts = np.zeros([tot_length // length_range[0] + 1, 2])
        path_dir = np.zeros([tot_length // length_range[0] + 1, 2])
        start_xy = np.array([0, 0])

        path_length = 0
        direction = 0
        path_pts[0] = start_xy
        point_i = 1
        while path_length < tot_length:
            length = random.randint(*length_range)
            path_length += length
            if (remaining := tot_length - path_length) < length_range[0]:
                length += remaining
                path_length += remaining

            d_direction = random.random() * abs(dir_range[1] - dir_range[0]) + dir_range[0]
            d_direction *= random.choice([-1, 1])
            direction += d_direction
            ex = np.cos(direction)
            ey = np.sin(direction)
            e = np.array([ex, ey])

            path_pts[point_i] = path_pts[point_i - 1] + e * length
            path_dir[point_i - 1] = np.array([length, direction])

            point_i += 1

        return path_pts[:point_i], path_dir[:point_i - 1]

    @staticmethod
    def create_road(path_dir, path_pts, width):
        half = width / 2
        normal_l = path_dir[..., 1] + np.pi / 2
        road_l = np.zeros_like(path_pts)
        road_r = road_l.copy()

        for i, p in enumerate(path_pts):
            if i == 0:
                road_l[i] = p + r_theta_2_x_y(np.array([half, normal_l[0]]))
                road_r[i] = p + r_theta_2_x_y(np.array([half, normal_l[0] + np.pi]))
            elif i == len(path_pts) - 1:
                road_l[i] = p + r_theta_2_x_y(np.array([half, normal_l[-1]]))
                road_r[i] = p + r_theta_2_x_y(np.array([half, normal_l[-1] + np.pi]))
            else:
                theta_2 = (path_dir[i, 1] - path_dir[i - 1, 1]) / 2
                d_length = np.arctan(theta_2) * half
                length = path_dir[i - 1, 0]
                direction = path_dir[i - 1, 1]
                road_l[i] = path_pts[i - 1] + r_theta_2_x_y(np.array([half, normal_l[i - 1]])) + r_theta_2_x_y(
                    np.array([length - d_length, direction]))
                road_r[i] = path_pts[i - 1] + r_theta_2_x_y(np.array([half, normal_l[i - 1] + np.pi])) + r_theta_2_x_y(
                    np.array([length + d_length, direction]))

        return np.vstack([road_l, road_r[::-1], road_l[0]])
