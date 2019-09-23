import cv2
import sys
import os
import shutil
import time
import numpy as np
import json
from multiprocessing import Process, Value
from openpose import pyopenpose as op
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame 

def set_params():
	# the parameters set for OpenPose
	params = dict()
	params["logging_level"] = 3
	params["output_resolution"] = "-1x-1"
	#params["net_resolution"] = "-1x368"
	params["net_resolution"] = "-1x224"
	params["model_pose"] = "BODY_25" # (BODY_25, COCO, MPI, MPI_4_layers).
	params["alpha_pose"] = 0.6
	params["scale_gap"] = 0.3
	params["scale_number"] = 1
	params["render_threshold"] = 0.05
	params["render_pose"] = 1
    # If GPU version is built, and multiple GPUs are available, set the ID here
	params["num_gpu_start"] = 0
	params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
	params["model_folder"] = "/home/xg/packages/openpose/models"
	return params

def write_video(path, im_pre, video_name, fps, size):
	'''
	Convert a serie of images to a video to the current path. The image sequence should be end with 
	a number in ascending order. 
	path: the folder path of the images
	im_pre: the prefix of the names of the images, eg, if the images are im1.jpg, im2.jpg ... im_pre = 'im'
	video_name: the name of the video to store
	fps: the output video's fps
	size: the H*W size of the images
	'''
	# total image/frame number of the images
	image_num = len([x for x in os.listdir(path)])
	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), float(fps), size, True)
	for i in range(image_num):
		im_name = os.path.join(path, im_pre) + str(i+1) + '.jpg'
		print(im_name)
		img = cv2.imread(im_name)
		video.write(img)
	video.release()

def Create_Refrash_Folder(path):
	'''
	Create an empty folder, if the folder is already existed, first remove it.
	'''
	if(os.path.exists(path)):
		shutil.rmtree(path)
	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)

def KinectStream(frame_count, BreakKinect, enable_rgb = True, enable_depth = True):
	# create folders for the color and depth images, respectively
	last_path, _ = os.path.split(os.getcwd())
	path_color = os.path.join(last_path, "color")
	path_depth = os.path.join(last_path, "depth")
	Create_Refrash_Folder(path_color)
	Create_Refrash_Folder(path_depth)	
	'''
	# if the depth images are needed, you must use OpenGLPacketPipeline to use GPU to
	# render it to achiece a real-time capture.
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
	SetSize = (512, 424)
	#SetSize = (1080, 720)

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
			registration.apply(color, depth, undistorted, registered, enable_filter = False)
		elif enable_depth:
 			registration.undistortDepth(depth, undistorted)

		if enable_rgb:
			start = time.time()
			new_frame = cv2.resize(color.asarray(), SetSize)
			new_frame = cv2.cvtColor(new_frame[:,:,:-1], cv2.COLOR_RGB2BGR)
			#new_frame = registered.asarray(np.uint8)
			#new_frame = cv2.cvtColor(new_frame[:,:,[0,1,2]], cv2.COLOR_RGB2BGR)
			cv2.imwrite(im_path_color, new_frame)
			print("Kinect color:", new_frame.shape)
		'''
		if enable_depth:
			#depth_frame = cv2.resize(depth.asarray()/ 4500., SetSize)
			depth = depth.asarray()/ 4500.
			cv2.imshow("color", new_frame)
			#cv2.imwrite(im_path_color, new_frame)
			print("Kinect depth:",depth.shape)
		'''
		listener.release(frames)

	device.stop()
	device.close()
	#write_video(path, im_pre = 'image', video_name = 'test.avi', fps = 15, size = SetSize)
	sys.exit(0)

def pose_estimation(frame_count, BreakKinect, UseDepth = False, Out_folder = 'Pose'):
	last_path, _ = os.path.split(os.getcwd())
	Read_folder_color = "color"
	Read_path = os.path.join(last_path, Read_folder_color)
	Write_path = os.path.join(last_path, Out_folder)
	Create_Refrash_Folder(Write_path)

	params = set_params()
    #Constructing OpenPose object allocates GPU memory
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()
	datum = op.Datum()
	local_frame = 0
	read_frame = 0
	while True:
		start = time.time()
		while(frame_count.value < read_frame + 2):
			a = 1
		load_time = time.time()
		local_frame = local_frame + 1
		read_frame = frame_count.value - 1
		file_name_in = 'image'+str(int(read_frame))+'.jpg'
		file_name_out = 'image'+str(int(local_frame))+'.jpg'
		im_path_in = os.path.join(Read_path, file_name_in)
		im_path_out = os.path.join(Write_path, file_name_out)
		frame = cv2.imread(im_path_in, 1)
		datum.cvInputData = frame
		opWrapper.emplaceAndPop([datum])
		out = datum.cvOutputData
		# print("Body keypoints: \n" + str(datum.poseKeypoints))
		print(datum.poseKeypoints)
		cv2.imshow('openpose_out', out)
		cv2.imwrite(im_path_out, out)
		print("Openpose:",time.time() - load_time, load_time - start)
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()
	write_video(Write_path, im_pre = 'image', video_name = '../test.avi', fps = 30, size = (512, 424))
	BreakKinect.value = 1


if __name__ == '__main__':
	# shared variables for the concurrent processes
	frame_count = Value("d", 0)
	BreakKinect = Value("d", 0)
	KN = Process(target=KinectStream, args=(frame_count, BreakKinect, True, True,))
	OP = Process(target=pose_estimation, args=(frame_count, BreakKinect, False, 'Pose', ))
	KN.start()
	OP.start()
	OP.join()
	KN.join()