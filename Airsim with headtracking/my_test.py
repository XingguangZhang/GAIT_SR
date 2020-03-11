# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

#import setup_path 
import airsim

import pprint
import os
import time
from math import sin

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()
'''
airsim.wait_key('Press any key to set camera-0 gimble to 15-degree pitch')
client.simSetCameraOrientation("0", airsim.to_quaternion(0.261799, 0, 0)); #radians

airsim.wait_key('Press any key to get camera parameters')
for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d: %s" % (camera_name, pp.pprint(camera_info)))
'''
airsim.wait_key('Press any key to move with a sin function')
for x in range(600): # do few times
    start = time.time()
    z = sin(x/10) - 13 # some random number
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x/20, 0, z/10), airsim.to_quaternion(0, 0, 0)), True)
    time.sleep(0.033-(time.time()-start))
    # print(time.time()-start)

airsim.wait_key('Press any key to reset')
# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetPose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)
