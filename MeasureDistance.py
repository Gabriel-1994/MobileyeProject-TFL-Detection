import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
#transform pixels into normalized pixels using the focal length and principle point
    new_pts = []
    for pt in pts:
        new_pts.append([(pt[0] - pp[0]) / focal, (pt[1] - pp[1]) / focal])
    new_pts = np.array(new_pts)
    return new_pts


def unnormalize(pts, focal, pp):
# transform normalized pixels into pixels using the focal length and principle point
    for pt in pts:
        pt[0] = (pt[0] * focal) + pp[0]
        pt[1] = (pt[1] * focal) + pp[1]
    return pts


def decompose(EM):
#extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    tz = EM[2, 3]
    if tz:
        foe = np.array([EM[0, 3], EM[1, 3]])/tz
    else:
        foe = []
    return R, foe, tz


def rotate(pts, R):
# rotate the points - pts using R
    rotated_pts = []
    for pt in pts:
        rotated_pt = R.dot(np.array([pt[0], pt[1], 1]))
        rotated_pt[0] = rotated_pt[0] / rotated_pt[2]
        rotated_pt[1] = rotated_pt[1] / rotated_pt[2]
        rotated_pts.append((rotated_pt[0], rotated_pt[1]))
    return np.array(rotated_pts)


def find_corresponding_points(p, norm_pts_rot, foe):
# compute the epipolar line between p and foe
# run over all norm_pts_rot and find the one closest to the epipolar line
# return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])
    min_dist = -1
    min_ind = -1
    for i in range(len(norm_pts_rot)):
        dist = abs(m * norm_pts_rot[i][0] + n - norm_pts_rot[i][1]) / np.sqrt(m * m + 1)
        min_dist, min_ind = check_if_min_distance_is_found(min_dist, dist, i, min_ind)
    return min_ind, norm_pts_rot[min_ind]


def check_if_min_distance_is_found(min_dist, curr_dist, index, min_index):
    if (min_dist == -1) or curr_dist < min_dist:
        min_dist = curr_dist
        min_index = index
    return min_dist, min_index



def calc_dist(p_curr, p_rot, foe, tZ):
# calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
# calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
# combine the two estimations and return estimated Z
    Zx = tZ*(foe[0] - p_rot[0])/(p_curr[0] - p_rot[0])
    Zy = tZ*(foe[1] - p_rot[1])/(p_curr[1] - p_rot[1])
    Zx_w = abs(p_curr[0] - p_rot[0])
    Zy_w = abs(p_curr[1] - p_rot[1])
    sum_w = Zx_w + Zy_w
    if (Zx_w + Zy_w) == 0:
        return 0
    Zx_w /= sum_w
    Zy_w /= sum_w
    Z = Zx_w*Zx + Zy_w*Zy
    return Z
