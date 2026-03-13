import math
import numpy as np


# distance helpers


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclidean_distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# vector helpers


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def dot_product(a, b):
    return np.dot(a, b)


# angle helpers


def angle_between(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def angle_3_points(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    return angle_between(ba, bc)


# clamp


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


# interpolation


def lerp(a, b, t):
    return a + (b - a) * t


# smoothing


def ema(prev, curr, alpha=0.6):
    return prev * (1 - alpha) + curr * alpha


# easing functions used for animations


def ease_out_quad(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def ease_in_quad(t: float) -> float:
    return t * t


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


# range mapping


def map_range(value, in_min, in_max, out_min, out_max):
    if in_max - in_min == 0:
        return out_min
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
