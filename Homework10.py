import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from matplotlib.animation import FuncAnimation
import threading
import imutils

global_exit = False

# video stream settings
FFMPEG_BIN = os.getcwd() + '\\ffmpeg\\ffmpeg.exe'  # using external tool to retrieve frames from http source (no rtsp)
# cam_url = 'https://eu5.camflg.com:5443/LiveApp/streams/015290301572984587686747.m3u8?token=undefined' # 1280x720
# cam_url = 'https://eu5.camflg.com:5443/LiveApp/streams/129763646084267746564546.m3u8?token=undefined'
cam_url = 'https://eu5.camflg.com:5443/LiveApp/streams/875330193606328046540894.m3u8?token=undefined' ## 1920x1080

video_size_w = 1920
video_size_h = 1080
video_channels = 3

target_video_size_w = 960
target_video_size_h = 540

# general cv options
font = cv2.FONT_HERSHEY_SIMPLEX
out_video = cv2.VideoWriter('d:\\detection_tracking8.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (target_video_size_w, target_video_size_h))


# opencv pedestrian detector initialization
NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

model_path = os.getcwd() + '\\datasets\\opencv_yolo\\'
labelsPath = model_path + 'coco.names'
LABELS = open(labelsPath).read().strip().split('\n')

weights_path = model_path + 'yolov4-tiny.weights'
config_path = model_path + 'yolov4-tiny.cfg'

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

# opening video stream from a public webcam
pipe = sp.Popen([FFMPEG_BIN, "-i", cam_url,
                 "-loglevel", "quiet",  # no text output
                 "-an",  # disable audio
                 "-f", "image2pipe",
                 "-pix_fmt", "rgb24",
                 "-vcodec", "rawvideo", "-"],
                stdin=sp.PIPE, stdout=sp.PIPE, bufsize=video_size_w * (video_size_h + 18) * video_channels)

# global video frame buffers and counters
raw_video_frames = []
processed_video_frames = []

raw_frame_counter = 0
processed_frame_counter = 0

tracker_update_interval = 100  # all trackers will be updated (detector will run) every XX frames specified in this setting

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
    global raw_frame_counter
    while True and not global_exit:
        start = time.perf_counter()
        raw_image = pipe.stdout.read(video_size_w * video_size_h * video_channels)
        image = np.frombuffer(raw_image, dtype='uint8')
        image = imutils.resize(image.reshape((video_size_h, video_size_w, video_channels)), target_video_size_w, target_video_size_h)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pipe.stdout.flush()
        raw_frame_counter += 1
        frame_buffer.append(image)
        if len(frame_buffer) > 1:
            frame_buffer.pop(0)  # clean up buffer, only keep the last frame
        processing_time = time.perf_counter() - start
        time.sleep(max(1 / 30 - processing_time, 0))
        total_time = time.perf_counter() - start
        # print(f'\rRetrieving {1 / total_time} frames/sec', end='')
    return True


# run frame buffering (the function above) in a separate thread
thread = threading.Thread(target=process_video, args=[raw_video_frames])
thread.start()


# detection and tracking here
def process_frame(input_frame_buffer, output_frame_buffer):

    def write_video(img):
        out_video.write(img)

    def update_trackers(trackers, img, color):
        for tracker in trackers:
            ok, tbox = tracker.update(img)
            x1, y1 = tbox[0], tbox[1]
            width, height = tbox[2], tbox[3]
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), color, 2)

    global processed_frame_counter
    # wait until first frame appears
    while not input_frame_buffer:
        time.sleep(0.1)

    # process frame and drop it to the output buffer
    while True and not global_exit:
        start = time.perf_counter()
        image = input_frame_buffer[0]

        # detect pedestrians on the video for every {tracker_update_interval} frame. Will work for the very first frame - index 0
        if processed_frame_counter % tracker_update_interval == 0:  #large number (rare update) will give us good example of how good tracker is
            # re-create all trackers
            CSRT_trackers = []
            KCF_trackers = []

            # running detection
            results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index('person'))
            for res in results:
                x1 = res[1][0]
                y1 = res[1][1]
                x2 = res[1][2]
                y2 = res[1][3]

                # we can draw a green rectangle to mark detector results (it good :) )
                #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                tbox = (x1, y1, x2 - x1, y2 - y1)

                if results.index(res) % 2 == 0:  # odd object number
                    tracker = cv2.TrackerCSRT_create()
                    ok = tracker.init(image, tbox)
                    CSRT_trackers.append(tracker)
                else:  # even object number
                    tracker = cv2.TrackerKCF_create()
                    ok = tracker.init(image, tbox)
                    KCF_trackers.append(tracker)

        #  NOTE: I've found that on the video all texts added to the image with cv2.putText() and all frames are converted from RGB to BRG
        #  pyplot shows all colors correctly

        update_trackers(CSRT_trackers, image, (255, 0, 0))  # RED for CSRT
        update_trackers(KCF_trackers, image, (0, 0, 255))  # BLUE for KCF

        cv2.putText(image, f'Frame: {processed_frame_counter}', (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Update trackers every {tracker_update_interval} frames', (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Total  {len(CSRT_trackers)+len(KCF_trackers)} active trackers', (10, 120), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'RED - CSRT', (target_video_size_w - 220, 40), font, 1, (255, 0, 0), 4, cv2.LINE_8)
        cv2.putText(image, 'BLUE - KCF', (target_video_size_w - 220, 80), font, 1, (0, 0, 255), 4, cv2.LINE_AA)

        output_frame_buffer.append(image)
        write_video(image)
        processed_frame_counter += 1
        # print('frame processed')
        if len(output_frame_buffer) > 1:
            output_frame_buffer.pop(0)
        processing_time = time.perf_counter() - start
        # print(f'\rProcessing {1/processing_time} frames/sec', end='')
        # print(f'\rProcessing frame {processed_frame_counter}', end='')
    return True


while not raw_video_frames:
    time.sleep(0.1)

# process image in a separate thread
thread_p = threading.Thread(target=process_frame, args=[raw_video_frames, processed_video_frames])
thread_p.start()

# wait until first processed video frame appears
while not processed_video_frames:
    time.sleep(0.1)

ax1 = plt.subplot(111)
im1 = ax1.imshow(processed_video_frames[0])


def update(i):
    if processed_video_frames:
        im1.set_data(processed_video_frames[0])


ani = FuncAnimation(plt.gcf(), update, interval=50)

plt.show()

global_exit = True
