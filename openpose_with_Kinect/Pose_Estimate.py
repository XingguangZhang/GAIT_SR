import cv2
import os
import time
import numpy as np
from support import WriteVideo, CreateRefrashFolder, WritePose, LiftPose
from multiprocessing import Process, Value, Pipe
from openpose import pyopenpose as op
from ReadKinect import KinectStream
from SkeShow import ShowSkeleton, ShowMat


def set_params():
    # the parameters set for OpenPose
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    # params["net_resolution"] = "-1x368"
    params["net_resolution"] = "-1x224"
    params["model_pose"] = "BODY_25"  # (BODY_25, COCO, MPI, MPI_4_layers).
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


def pose_estimation(sk_s, frame_count, BreakKinect, UseDepth=False, Out_folder='Pose'):
    last_path, _ = os.path.split(os.getcwd())
    Read_folder_color = "color"
    Read_folder_depth = "depth"
    Read_path = os.path.join(last_path, Read_folder_color)
    Read_path_depth = os.path.join(last_path, Read_folder_depth)
    Write_path = os.path.join(last_path, Out_folder)
    CreateRefrashFolder(Write_path)

    params = set_params()
    # Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()
    local_frame = 0
    read_frame = 0
    lp = LiftPose((512, 424))
    while True:
        start = time.time()
        # wait if there's no new frame
        while frame_count.value < read_frame + 2:
            a = 1
        load_time = time.time()
        local_frame = int(local_frame) + 1
        read_frame = int(frame_count.value) - 1
        file_name_in = 'image' + str(read_frame) + '.jpg'
        file_name_out = 'image' + str(local_frame) + '.jpg'
        im_path_in = os.path.join(Read_path, file_name_in)
        im_path_depth = os.path.join(Read_path_depth, file_name_in)
        im_path_out = os.path.join(Write_path, file_name_out)
        frame = cv2.imread(im_path_in, 1)
        depth_frame = cv2.imread(im_path_depth, 0)
        # start to use openpose python wrapper
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        out_im = datum.cvOutputData
        # print(datum.poseKeypoints.shape)

        if datum.poseKeypoints.size >= 75:
            # print(datum.poseKeypoints.shape)
            pos3D = lp.LiftTo3D(datum.poseKeypoints, depth_frame)
            sk_s.send(pos3D)
            WritePose(pos3D, "poseCor.txt")
        cv2.imshow('openpose_out', out_im)
        cv2.imwrite(im_path_out, out_im)

        print("Openpose:", time.time() - load_time, load_time - start, frame.shape)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

    WriteVideo(Write_path, im_pre='image', video_name='../test.avi', fps=27, size=(540, 360))
    BreakKinect.value = 1
    sk_s.send("Break")
    print("Kinect closed!")

if __name__ == '__main__':
    # shared variables for the concurrent processes
    frame_count = Value("d", 0)
    BreakKinect = Value("d", 0)
    sk_s, sk_r = Pipe()
    KN = Process(target=KinectStream, args=(frame_count, BreakKinect, True, True,))
    OP = Process(target=pose_estimation, args=(sk_s, frame_count, BreakKinect, False, 'Pose',))
    SS = Process(target=ShowSkeleton, args=(sk_r,))
    KN.start()
    OP.start()
    SS.start()
    SS.join()
    OP.join()
    KN.join()
