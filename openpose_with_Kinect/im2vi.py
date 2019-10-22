import cv2

fps = 15
video = cv2.VideoWriter('record.avi',cv2.VideoWriter_fourcc(*'MJPG'), float(fps), (576,324), True)

for i in range(50):
    im_name = 'images/image' + str(i+1) + '.jpg'
    print(im_name)
    img = cv2.imread(im_name)
    video.write(img)
video.release()