import cv2
import os
import shutil
import numpy as np


def WriteVideo(path, im_pre, video_name, fps, size):
    """
    description:
    Convert a serie of images to a video to the current path. The image sequence should be end with
    a number in ascending order.
    input:
    path: the folder path of the images
    im_pre: the prefix of the names of the images, eg, if the images are im1.jpg, im2.jpg ... im_pre = 'im'
    video_name: the name of the video to store
    fps: the output video's fps
    size: the H*W size of the images
    """
    # total image/frame number of the images
    image_num = len([x for x in os.listdir(path)])
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), float(fps), size, True)
    for i in range(image_num):
        im_name = os.path.join(path, im_pre) + str(i + 1) + '.jpg'
        print(im_name)
        img = cv2.imread(im_name)
        video.write(img)
    video.release()


def CreateRefrashFolder(path):
    """
    Create an empty folder, if the folder is already existed, first remove it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


class LiftPose(object):
    def __init__(self, imsize):
        self.jointarea = "jointArea.txt"
        self.PosList = []
        self.PosListLen = 50
        self.Height = imsize[1]
        self.Width = imsize[0]
        self.last_pos = []

    @staticmethod
    def FilterDepth(depth_im):
        kernel = np.ones((5, 5), np.uint8)
        depth_im = cv2.medianBlur(depth_im, 9)
        depth_im = cv2.dilate(depth_im, kernel, iterations=1)
        depth_im = cv2.erode(depth_im, kernel, iterations=2)
        return depth_im

    def FindDepth(self, x, y, depth_im, neighbor=0):
        assert neighbor >= 0
        valid = 5
        if neighbor == 0:
            return depth_im[x, y]
        else:
            depth_list = []
            for i in range(x - neighbor, x + neighbor + 1):
                for j in range(y - neighbor, y + neighbor + 1):
                    if 0 <= i < self.Height and 0 <= j < self.Width:
                        if depth_im[i, j] >= 10:
                            depth_list.append(depth_im[i, j])
            depth_list.sort()
            # print("depth_list: ", depth_list)
            if not depth_list:
                return 0
            else:
                return sum(depth_list[:valid]) / len(depth_list[:valid])

    @staticmethod
    def PosFilter(current, last):
        if len(last) == 25:
            for i in range(len(current)):
                (x, y, z) = current[i]
                (lx, ly, lz) = last[i]
                if x < 1 and lx > 1:
                    # if (x < 1 or abs(x - lx) >= 50) and lx > 1:
                    x = lx
                if y < 1 and ly > 1:
                    # if (y < 1 or abs(y - ly) >= 50) and ly > 1:
                    y = ly
                if (abs(z) < 10 or abs(z - lz) >= 60) and lz > 10:
                    z = lz
                current[i] = (x, y, z)
        return current

    def LiftTo3D(self, joints, depth_im):
        if joints.size >= 75:
            Pos3D = []
            depth_im = self.FilterDepth(depth_im)
            joints = joints[0, :, :]
            for i in range(25):
                x, y = joints[i, 0], joints[i, 1]
                w_pos = self.Width - 1 if y >= self.Width else int(y - 1)
                h_pos = self.Height - 1 if x >= self.Height else int(x - 1)
                d = self.FindDepth(w_pos, h_pos, depth_im, neighbor=3)
                '''
                x = (x-212)/212
                y = (y-256)/256
                d = (d - 3000) / 1500

                pos3D.append((2*x,-2*y,2*d))
                '''
                Pos3D.append((h_pos, w_pos, d))
            # print(Pos3D)
            Pos3D = self.PosFilter(Pos3D, self.last_pos)
            # print('Filtered:', Pos3D)
            self.PosList.append(Pos3D)
            self.last_pos = Pos3D
            if len(self.PosList) > self.PosListLen:
                self.PosList.pop(0)
            return Pos3D
        else:
            return 0


def WritePose(data, file):
    L = ""
    with open(file, 'a') as f:
        for pos in data:
            for cor in pos:
                L = L + str(cor) + ","
            L = L + ";"
        f.write(L + "\n")
        f.close()


def DrawOnIm(im, posList):
    radius = 3
    thickness = -1
    for p in posList[:15]:
        cor = (p[0], p[1])
        im = cv2.circle(im, cor, radius, 255, thickness)
    return im
