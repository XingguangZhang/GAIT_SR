import os
import time
from support import CreateRefrashFolder
import cv2
import numpy as np
from pykalman import KalmanFilter


def findspots(im, mask, spot_area_max, spot_area_min):
    spot_list = []
    num, labels = cv2.connectedComponents(mask)
    for i in range(num):
        (x, y) = np.where(labels == i)
        if spot_area_min <= x.shape[0] <= spot_area_max:
            x_s = 0
            y_s = 0
            w_s = 0
            for j in range(x.shape[0]):
                w_s += im[x[j], y[j]]
                x_s += x[j] * im[x[j], y[j]]
                y_s += y[j] * im[x[j], y[j]]
            x_avg = round(x_s / w_s, 2)
            y_avg = round(y_s / w_s, 2)
            spot_list.append((x_s / w_s, y_s / w_s))
            # cv2.putText(im, str(i), (int(y_avg)+10, int(x_avg)+10), cv2.FONT_HERSHEY_SIMPLEX,0.35,255)
            # cv2.putText(im, str(x_avg)+' , '+str(y_avg), (int(y_avg)+10, int(x_avg)+10), cv2.FONT_HERSHEY_SIMPLEX,0.35,255)
            cv2.line(im, (int(y_avg) - 5, int(x_avg) - 5), (int(y_avg) + 5, int(x_avg) + 5), 255, 1)
            cv2.line(im, (int(y_avg) - 5, int(x_avg) + 5), (int(y_avg) + 5, int(x_avg) - 5), 255, 1)
    return im, spot_list


def track_spots(im, spot_list, spots_num, prev_spot_list):
    def takeSecond(elem):
        return elem[1]

    # pose_list = []
    if not prev_spot_list:
        prev_spot_list.append(spot_list.pop(0))
        spot_list.sort(key=takeSecond)
        for i in range(spots_num - 1):
            prev_spot_list.append(spot_list.pop(0))

    else:
        sv = np.asarray(spot_list)
        sv = sv - np.average(sv, axis=0)
        psv = np.asarray(prev_spot_list)
        psv = psv - np.average(psv, axis=0)
        x_psv, x_sv = np.meshgrid(psv[:, 0], sv[:, 0])
        x_diff = x_psv - x_sv
        y_psv, y_sv = np.meshgrid(psv[:, 1], sv[:, 1])
        y_diff = y_psv - y_sv
        diff_norm = x_diff ** 2 + y_diff ** 2
        indices = np.argmin(diff_norm, axis=0)
        prev_spot_list = []
        for i in range(spots_num):
            prev_spot_list.append(spot_list[indices[i]])
            x_avg, y_avg = prev_spot_list[i]
            cv2.putText(im, str(i + 1), (int(y_avg) + 10, int(x_avg) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)

    return im, prev_spot_list


class spotfilter(object):
    def __init__(self, spot_num):
        self.A = [[1, 0], [0, 1]]
        self.C = [[1, 0],
                  [0, 1]]
        self.Q = np.asarray([[1e-2, 1e-3],
                             [1e-3, 1e-2]])
        self.R = np.asarray([[1, 1e-3],
                             [1e-3, 1]]) / 10
        self.num_filter = spot_num
        self.KF = []
        self.state_mean_list = [np.zeros(2) for i in range(spot_num)]
        self.state_covariance_list = [np.asarray([[1, 0], [0, 1]]) for i in range(spot_num)]

    def init_Kalman(self, init_list):
        mu0 = np.asarray(init_list)
        for i in range(self.num_filter):
            self.KF.append(KalmanFilter(transition_matrices=self.A,
                                        observation_matrices=self.C,
                                        transition_covariance=self.Q,
                                        observation_covariance=self.R,
                                        initial_state_mean=mu0[i, :]))

    def update(self, spot_list):
        measurement = np.asarray(spot_list)
        new_spot_list = []
        for i in range(self.num_filter):
            self.state_mean_list[i], self.state_covariance_list[i] = self.KF[i].filter_update(
                filtered_state_mean=self.state_mean_list[i],
                filtered_state_covariance=self.state_covariance_list[i],
                observation=measurement[i, :])
            new_spot_list.append((self.state_mean_list[i][0], self.state_mean_list[i][1]))
        return new_spot_list


class anglefilter(object):
    def __init__(self):
        self.A_a = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        self.C_a = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        self.Q_a = np.asarray([[1e-3, 1e-4, 1e-4],
                               [1e-4, 1e-2, 1e-4],
                               [1e-4, 1e-4, 1e-2]]) / 50
        self.R_a = np.asarray([[0.1, 1e-3, 1e-3],
                               [1e-3, 0.1, 1e-3],
                               [1e-3, 1e-3, 0.1]]) / 50
        self.state_mean = np.zeros(3)
        self.state_covariance = self.Q_a
        self.filter = KalmanFilter(transition_matrices=self.A_a,
                                   observation_matrices=self.C_a,
                                   transition_covariance=self.Q_a,
                                   observation_covariance=self.R_a,
                                   initial_state_mean=np.asarray([0, 0, 0]))

    def update(self, angle_list):
        measurement = np.asarray(angle_list[:3])
        self.state_mean, self.state_covariance = self.filter.filter_update(filtered_state_mean=self.state_mean,
                                                                           filtered_state_covariance=self.state_covariance,
                                                                           observation=measurement)
        new_spot_list = [self.state_mean[0], self.state_mean[1], self.state_mean[2]] + angle_list[3:]
        return new_spot_list


def comp_angles_Alter(prev_spot_list):
    R_13, R_01, R_03 = 170 * 3, 143 * 3, 145 * 3
    spot_vec = np.asarray(prev_spot_list)  # (l->r, top->bottom)
    c_13 = np.linalg.norm((spot_vec[1, :] + spot_vec[3, :]) / 2 - spot_vec[0, :])
    d_13_vec = spot_vec[3, :] - spot_vec[1, :]
    roll = np.arctan(d_13_vec[0] / d_13_vec[1])

    d_13 = np.linalg.norm(d_13_vec)
    d_01 = np.linalg.norm(spot_vec[1, :] - spot_vec[0, :])
    d_03 = np.linalg.norm(spot_vec[3, :] - spot_vec[0, :])
    a = (R_13 + R_01 + R_03) * (-R_01 + R_03 + R_13) * (R_01 - R_03 + R_13) * (R_01 + R_03 - R_13)
    c = (d_13 + d_01 + d_03) * (-d_01 + d_03 + d_13) * (d_01 - d_03 + d_13) * (d_01 + d_03 - d_13)
    b = d_01 ** 2 * (-R_01 ** 2 + R_03 ** 2 + R_13 ** 2) + d_03 ** 2 * (
                R_01 ** 2 - R_03 ** 2 + R_13 ** 2) + d_13 ** 2 * (R_01 ** 2 + R_03 ** 2 - R_13 ** 2)
    s = np.sqrt((b + np.sqrt(b ** 2 - a * c)) / a)
    sigma = 1 if (d_01 ** 2 + d_03 ** 2 - d_13 ** 2) <= s ** 2 * (R_01 ** 2 + R_03 ** 2 - R_13 ** 2) else -1
    h_1, h_3 = np.sqrt((s * R_01) ** 2 - d_01 ** 2), sigma * np.sqrt((s * R_03) ** 2 - d_03 ** 2)
    h_c13 = (h_1 + h_3) / 2

    yaw = np.arctan((h_3 - h_1) / d_13)
    pitch = np.arctan(h_c13 / c_13) - np.arctan(5 / 11.5)

    return [roll, yaw, pitch, h_1 / s, h_3 / s]


def euler_to_quaternion(roll, yaw, pitch):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


def quaternion_multiplication(first, second):
    [w1, x1, y1, z1] = first
    [w2, x2, y2, z2] = second
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + z1 * y2 - y1 * z2
    y = w1 * y2 + y1 * w2 + x1 * z2 - z1 * x2
    z = w1 * z2 + z1 * w2 + y1 * x2 - x1 * y2
    return [w, x, y, z]


def HPE(BreakKinect, W, X, Y, Z):

    capture = cv2.VideoCapture(0)
    '''
    try:
        capture = cv2.VideoCapture(0)
    except:
        print("Open video 0 failed")
        capture = cv2.VideoCapture(1)
    '''

    out_path = '/home/xg/Documents/gait/cap'
    CreateRefrashFolder(out_path)
    idx = 0

    spot_area_max = 300
    spot_area_min = 5
    spots_num = 4
    kernel = np.ones((2, 2), np.uint8)
    prev_spot_list = []
    angle_list = [0, 0, 0]
    spot_filter = spotfilter(spots_num)
    angle_filter = anglefilter()

    filename = '/home/xg/Documents/gait/codes/posRecord_124.txt'
    with open(filename, 'r') as f:
        content = f.readlines()
        num = len(content)

    while not BreakKinect.value:  # do few times
        for i in range(num):
            line = content[i * 2]
            idx += 1
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # MAX_BRIGHTNESS = 2 * np.average(gray)
            MAX_BRIGHTNESS = 100
            im = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
            mask = np.where(im > MAX_BRIGHTNESS, im, 0)
            # mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            im, spot_list = findspots(im, mask, spot_area_max, spot_area_min)
            if len(spot_list) == spots_num:
                im, pev_spot_list = track_spots(im, spot_list, spots_num, prev_spot_list)
                '''
                if spot_filter.KF == []:
                    spot_filter.init_Kalman(pev_spot_list)
                else:
                    pev_spot_list = spot_filter.update(pev_spot_list)
                '''

                # angle_list = comp_angles(pev_spot_list)
                # print('my:',comp_angles(pev_spot_list))
                angle_list = comp_angles_Alter(pev_spot_list)
                # print('Alter:',angle_list)

                angle_list = angle_filter.update(angle_list)

            roll, yaw, pitch = (angle_list[0], 5 * angle_list[1], 2 * angle_list[2])
            quat_list = euler_to_quaternion(roll, yaw, pitch)
            write_name = os.path.join(out_path, str(idx) + '.jpg')
            line = line.strip()
            t, x, y, z, w, qx, qy, qz = line.split(' ')
            t = float(t)
            x = float(x)
            y = float(y)
            z = float(z)

            w = float(w)
            qx = float(qx)  # roll
            qy = float(qy)  # pitch + up
            qz = float(qz)  # yaw + right
            [new_w, new_x, new_y, new_z] = quaternion_multiplication(quat_list, [w, qx, qy, qz])
            #HeadPose = [new_w, new_x, new_y, new_z]
            #print('HeadPose:', HeadPose)
            [W.value, X.value, Y.value, Z.value] = quat_list
            # W.value, X.value, Y.value, Z.value = new_w, new_x, new_y, new_z
            cv2.imwrite(write_name, im)
            time.sleep(0.002)
            cv2.imshow('frame', im)
            cv2.waitKey(1)
            if BreakKinect.value:
                break
