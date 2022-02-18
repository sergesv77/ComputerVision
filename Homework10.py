import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from matplotlib.animation import FuncAnimation
import threading
import imutils

FFMPEG_BIN = os.getcwd() + '\\ffmpeg\\ffmpeg.exe'
cam_url = 'https://eu5.camflg.com:5443/LiveApp/streams/015290301572984587686747.m3u8?token=undefined'

video_size_w = 1280
video_size_h = 720
video_channels = 3

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

model_path = os.getcwd() + '\\datasets\\opencv_yolo\\'
labelsPath = model_path + 'coco.names'
LABELS = open(labelsPath).read().strip().split('\n')

weights_path = model_path + 'yolov4-tiny.weights'
config_path = model_path + 'yolov4-tiny.cfg'

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i-1] for i in model.getUnconnectedOutLayers()]

# opening video stream from a public webcam
pipe = sp.Popen([FFMPEG_BIN, "-i", cam_url,
                 "-loglevel", "quiet",  # no text output
                 "-an",  # disable audio
                 "-f", "image2pipe",
                 "-pix_fmt", "rgb24",
                 "-vcodec", "rawvideo", "-"],
                stdin=sp.PIPE, stdout=sp.PIPE, bufsize=video_size_w * (video_size_h + 18) * video_channels)

# global video frame buffer
video_frames = []

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    return results


# continuous buffering - collect frames as fast as we can
def process_video(frame_buffer):
    while True:
        raw_image = pipe.stdout.read(video_size_w * video_size_h * video_channels)
        image = np.frombuffer(raw_image, dtype='uint8')
        image = image.reshape((video_size_h, video_size_w, video_channels))
        pipe.stdout.flush()
        frame_buffer.append(image)
        if len(frame_buffer) > 1:
            frame_buffer.pop(0)  # clean up buffer, only keep the last frame


# run frame buffering (the function above) in a separate thread
thread = threading.Thread(target=process_video, args=[video_frames])
thread.start()


# retrieve next frame from buffer once we are ready to process it
def GetFrame(frame_buffer, wait_timeout=20):
    original_image = None
    waiting = 0
    wait_interval = 0.1
    while not frame_buffer and waiting < wait_timeout:
        time.sleep(wait_interval)
        waiting += wait_interval
    if frame_buffer:
        original_image = frame_buffer[0]
    return original_image


# Run detection and tracking here
def process_frame(image):
    # A not-very-good opencv pedestrian detector
    # img_result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # (regions, _) = hog.detectMultiScale(image,
    #                                     winStride=(4, 4),
    #                                     padding=(8, 8),
    #                                     scale=0.95)
    # for (x, y, w, h) in regions:
    #     cv2.rectangle(image, (x, y),
    #                   (x + w, y + h),
    #                   (0, 0, 255), 2)

    # img_copy = imutils.resize(image, width=700)
    # img_copy = np.copy(image)
    results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index('person'))
    for res in results:
        cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

    return image


ax1 = plt.subplot(111)

# create axes
im1 = ax1.imshow(process_frame(GetFrame(video_frames)))


def update(i):
    im1.set_data(process_frame(GetFrame(video_frames)))


ani = FuncAnimation(plt.gcf(), update, interval=50)

plt.show()

# try:
#     while video.isOpened():
#         ret, frame = video.read()
#         cv2.imshow('video', frame)
#         time.sleep(5)
# finally:
#     video.release()
