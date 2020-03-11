# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

# import setup_path
import airsim

import pprint
import os
import cv2
from math import sin


def AirView(W, X, Y, Z, skeleton_recv, dir=1, rate = 1):
    pp = pprint.PrettyPrinter(indent=4)

    client = airsim.VehicleClient()
    client.confirmConnection()

    # airsim.wait_key('Press any key to start the tracking')

    x_init, y_init, z_init = 0, 0, -1.6
    while True:

        sk = skeleton_recv.recv()
        # print('AirSimCV received:', sk)
        if isinstance(sk, list) and len(sk) == 25:
            FacePos = sk[0]
            x_shift = - FacePos[2] / 50
            y_shift = FacePos[0] / 212
            z_shift = FacePos[1] / 256
            #print("received HeadPose:", HeadPose[0])
            n_w, n_qx, n_qy, n_qz = W.value, X.value, Y.value, Z.value
            if dir:
                client.simSetVehiclePose(
                    airsim.Pose(airsim.Vector3r(x_init + x_shift, y_init + rate*y_shift, z_init + rate*z_shift),
                                airsim.Quaternionr(n_qx, n_qy, n_qz, n_w)), True)
            else:
                client.simSetVehiclePose(
                    airsim.Pose(airsim.Vector3r(x_init - x_shift, y_init - rate*y_shift, z_init - rate*z_shift),
                                airsim.Quaternionr(n_qx, n_qy, n_qz, n_w)), True)

        elif sk == "Break":
            print("Tracking terminating...")
            client.simSetPose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0.0, 0, 0)), True)
            break
        elif sk == 'Empty':
            i = 1
    client.simSetPose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)


'''
def AirView(skeleton_recv, dir=1, rate=1):
    pp = pprint.PrettyPrinter(indent=4)

    client = airsim.VehicleClient()
    client.confirmConnection()
    
    # airsim.wait_key('Press any key to start the tracking')
    f = open('/home/xg/packages/track/posRecord_124.txt', 'r')
    x_init, y_init, z_init = 0, 0, -1.6
    while True:
        for line in f.readlines():
            sk = skeleton_recv.recv()
            line = line.strip()
            print(line)
            t, xx, yy, zz, w, qx, qy, qz = line.split(' ')
            t = float(t)
            xx = float(xx)
            yy = float(yy)
            zz = float(zz)
            w = float(w)
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)

            if isinstance(sk, list) and len(sk) == 25:
                FacePos = sk[0]
                x_shift = - FacePos[2] / 50
                y_shift = FacePos[0] / 212
                z_shift = FacePos[1] / 256
                if dir:
                    client.simSetVehiclePose(
                        airsim.Pose(airsim.Vector3r(x_init + x_shift + xx, yy + y_init + rate * y_shift, zz + z_init + rate * z_shift),
                                    airsim.to_quaternion(0, 0, 0)), True)
                else:
                    client.simSetVehiclePose(
                        airsim.Pose(airsim.Vector3r(x_init - x_shift + xx, yy + y_init - rate * y_shift, zz + z_init - rate * z_shift),
                                    airsim.to_quaternion(0, 0, 0)), True)
            if sk == "Break":
                print("Tracking terminating...")
                break

        client.simSetPose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)
'''