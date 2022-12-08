from init import *


def rad_norm(angle):
    return angle % (2 * np.pi) / np.pi


def above_pi(angle):
    return 1 if 1 < rad_norm(angle) <= 2 else -1


def is_positive(num):
    return 1 if num > 0 else 0


def is_negative(num):
    return 1 if num <= 0 else 0


# 求向量模
def mod(vec: np.ndarray):
    return np.sqrt(np.sum(vec ** 2, axis=-1))


# 求向量方向的单位向量
def dir_e(vec: np.ndarray):
    return vec / mod(vec)


# 求向量角度的单位向量
def theta_e(theta):
    if isinstance(theta, np.ndarray):
        ex = np.cos(theta)
        ey = np.sin(theta)
        return np.hstack([ex, ey])
    return np.array([np.cos(theta), np.sin(theta)])


# 求二维平面内向量的法线方向（左手方向）
def normal_2d(vec: np.ndarray):
    assert vec.shape[-1] == 2, "函数只接受二维向量"
    x, y = -vec[..., 1], vec[..., 0]
    return dir_e(np.hstack([x, y]))


# 求两个向量的夹角,角的大小使用弧度表示.（可以使用余弦公式快速求得） cos(theta) = (a dot b) / (mod(a) * mod(b))
def d_theta(vec_a: np.ndarray, vec_b: np.ndarray):
    cos_theta = np.dot(vec_a, vec_b) / (mod(vec_a) * mod(vec_b))
    if isinstance(cos_theta, np.ndarray):
        cos_theta[cos_theta >= 1.0] = 1.0
        cos_theta[cos_theta <= -1.0] = -1.0
    else:
        cos_theta = min(1.0, max(-1, cos_theta))
    return np.arccos(cos_theta)


# 求向量的方向角theta
def dir2theta(direction):
    direction = dir_e(direction)
    cos_dir, sin_dir = direction[0], direction[1]
    if np.equal(cos_dir, 0):
        return direction[1] * np.pi / 2
    else:
        return np.arctan(sin_dir / cos_dir) + is_negative(cos_dir) * np.pi


# 笛卡尔2极坐标
def x_y_2_r_theta(vec: np.ndarray):
    x, y = vec[..., 0], vec[..., 1]
    r = mod(vec[..., :2])
    theta = np.arctan(y / x)
    return np.vstack([r, theta]).T


# 极坐标2笛卡尔
def r_theta_2_x_y(vec: np.ndarray):
    r, theta = vec[..., 0], vec[..., 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y]).T


# 判断是否在凸多边形内(pts的点顺序必须沿顺时针或逆时针，不可乱序)
def in_convex_polygon(p, pts):
    dirs = pts - p
    ang = 0
    for i in range(len(dirs)):
        ang += d_theta(dirs[i], dirs[i - 1])
    return np.abs(ang - 2 * np.pi) < 1e-7


# 求点在线段上投影的位置（距离line上0点的距离）
def pos_normal(p, line):
    a, b = line[0], line[1]
    ap, pb, ab = p - a, b - p, b - a
    length = mod(ab)

    theta_1 = d_theta(ap, ab)
    theta_2 = d_theta(pb, ab)
    if theta_2 == 0:
        return mod(ap)
    tan1, tan2 = np.tan(theta_1), np.tan(theta_2)

    return length / (1 + (tan1 / tan2))


# 快速排斥实验
def rect_cross(p1, p2, q1, q2):
    ax, ay = p1
    bx, by = p2
    cx, cy = q1
    dx, dy = q2
    not_ret = max(cx, dx) < min(ax, bx) or \
              max(cy, dy) < min(ay, by) or \
              max(ax, bx) < min(cx, dx) or \
              max(ay, by) < min(cy, dy)

    return not not_ret


# 交点计算
def point_cross(p1, p2, q1, q2):
    if not rect_cross(p1, p2, q1, q2):
        return False, np.array([0, 0])

    dp = p2 - p1
    dq = q2 - q1

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1

    b1 = dp[1] * x1 - dp[0] * y1
    b2 = dq[1] * x3 - dq[0] * y3

    d = dp[0] * dq[1] - dq[0] * dp[1]
    if d == 0:
        return False, np.array([0, 0])

    d1 = b2 * dp[0] - b1 * dq[0]
    d2 = b2 * dp[1] - b1 * dq[1]

    x0 = d1 / d

    if min(x1, x2) <= x0 <= max(x1, x2):
        return True, np.array([x0, d2 / d])
    return False, np.array([0, 0])
