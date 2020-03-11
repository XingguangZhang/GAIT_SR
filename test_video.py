import cv2
from openpose import pyopenpose as op


def set_params():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["model_folder"] = "/home/xg/packages/openpose/models"
    return params


def main():
    params = set_params()

    # Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Opening OpenCV stream
    stream = cv2.VideoCapture('test_turn.avi')

    font = cv2.FONT_HERSHEY_SIMPLEX
    datum = op.Datum()
    while True:

        ret, img = stream.read()
        # Process Image
        cv2.imshow('original', img)
        print(img.shape)
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        output_image = datum.cvOutputData

        # Display the stream
        cv2.putText(output_image, 'OpenPose using Python-OpenCV', (20, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Human Pose Estimation', output_image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
