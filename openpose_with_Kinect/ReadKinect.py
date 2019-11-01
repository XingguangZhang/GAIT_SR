from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
import numpy as np
import os
from support import CreateRefrashFolder
import time
import cv2
import sys


def KinectStream(frame_count, BreakKinect, enable_rgb=True, enable_depth=True):
    # create folders for the color and depth images, respectively
    last_path, _ = os.path.split(os.getcwd())
    path_color = os.path.join(last_path, "color")
    path_depth = os.path.join(last_path, "depth")
    CreateRefrashFolder(path_color)
    CreateRefrashFolder(path_depth)
    '''
	# if the depth images are needed, you must use OpenGLPacketPipeline for enabling GPU to
	# render the depth map for so that the real-time capture can be achieved.
	try:
		from pylibfreenect2 import OpenGLPacketPipeline
		pipeline = OpenGLPacketPipeline()
	except:
		try:
			from pylibfreenect2 import OpenCLPacketPipeline
			pipeline = OpenCLPacketPipeline()
		except:
			from pylibfreenect2 import CpuPacketPipeline
			pipeline = CpuPacketPipeline()
	'''

    if enable_depth:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    else:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    types = 0
    if enable_rgb:
        types |= FrameType.Color
    if enable_depth:
        types |= (FrameType.Ir | FrameType.Depth)
    listener = SyncMultiFrameListener(types)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    if enable_rgb and enable_depth:
        device.start()
    else:
        device.startStreams(rgb=enable_rgb, depth=enable_depth)
    # NOTE: must be called after device.start()
    if enable_depth:
        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    # the target size of the resize function
    SetSize = (540, 360)
    # SetSize = (1080, 720)

    while not BreakKinect.value:
        frame_count.value += 1
        file_name = 'image' + str(int(frame_count.value)) + '.jpg'
        im_path_color = os.path.join(path_color, file_name)
        im_path_depth = os.path.join(path_depth, file_name)
        frames = listener.waitForNewFrame()

        if enable_rgb:
            color = frames["color"]
        if enable_depth:
            ir = frames["ir"]
            depth = frames["depth"]

        if enable_rgb and enable_depth:
            registration.apply(color, depth, undistorted, registered, enable_filter=False)
        elif enable_depth:
            registration.undistortDepth(depth, undistorted)

        if enable_rgb:
            start = time.time()
            new_frame = cv2.resize(color.asarray(), SetSize)
            # new_frame = new_frame[:,:,:-1]
            # new_frame = cv2.cvtColor(new_frame[:,:,:-1], cv2.COLOR_RGB2BGR)
            new_frame = registered.asarray(np.uint8)
            # new_frame = cv2.cvtColor(new_frame[:,:,[0,1,2]], cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_path_color, new_frame[:, :, :-1])
        # print("Kinect color:", new_frame.shape)

        if enable_depth:
            # depth_frame = cv2.resize(depth.asarray()/ 4500., SetSize)
            depth = depth.asarray() / 4500.
            # cv2.imshow("color", new_frame)
            undist = undistorted.asarray(np.float32) / 4500 * 255
            cv2.imwrite(im_path_depth, undist.astype(np.uint8))
            print("Kinect depth:", depth.shape)

        listener.release(frames)

    device.stop()
    device.close()
    # WriteVideo(path, im_pre = 'image', video_name = 'test.avi', fps = 15, size = SetSize)
    sys.exit(0)
