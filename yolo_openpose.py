import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="YOLOv4 + OpenPose pipeline")
parser.add_argument('--image', required=True, help='Path to input image')
parser.add_argument('--device', default='cpu', help='cpu or gpu')
parser.add_argument('--yolo_cfg', default='yolo/yolov4.cfg')
parser.add_argument('--yolo_weights', default='yolo/yolov4.weights')
parser.add_argument('--pose_proto', default='coco/pose_deploy_linevec.prototxt')
parser.add_argument('--pose_weights', default='pose/coco/pose_iter_440000.caffemodel')
args = parser.parse_args()

# load networks
net_yolo = cv2.dnn.readNet(args.yolo_cfg, args.yolo_weights)
net_pose = cv2.dnn.readNetFromCaffe(args.pose_proto, args.pose_weights)

if args.device == 'gpu':
    net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net_pose.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_pose.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net_pose.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

# constants
inpWidth = 608
inpHeight = 608
pose_in_w = 368
pose_in_h = 368
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],
               [8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],
               [14,16],[15,17]]

# read image
frame = cv2.imread(args.image)
if frame is None:
    raise ValueError('Image not found')
height, width = frame.shape[:2]

# YOLO detection
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (inpWidth, inpHeight), swapRB=True, crop=False)
net_yolo.setInput(blob)
layer_names = net_yolo.getLayerNames()
output_layers = [layer_names[i-1] for i in net_yolo.getUnconnectedOutLayers().flatten()]
detections = net_yolo.forward(output_layers)

boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in detections:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if class_id == 0 and confidence > conf_threshold:  # person class
            box = detection[0:4] * np.array([width, height, width, height])
            centerX, centerY, w, h = box.astype('int')
            x = int(centerX - w/2)
            y = int(centerY - h/2)
            boxes.append([x, y, int(w), int(h), float(confidence)])

indices = cv2.dnn.NMSBoxes([b[:4] for b in boxes], [b[4] for b in boxes], conf_threshold, nms_threshold)

points_all = []
for i in indices.flatten():
    x, y, w, h, conf = boxes[i]
    roi = frame[max(0,y):min(y+h,height), max(0,x):min(x+w,width)]
    if roi.size == 0:
        continue
    inp = cv2.dnn.blobFromImage(roi, 1.0/255, (pose_in_w, pose_in_h), (0,0,0), swapRB=False, crop=False)
    net_pose.setInput(inp)
    output = net_pose.forward()
    H = output.shape[2]
    W = output.shape[3]
    points = []
    for j in range(nPoints):
        probMap = output[0, j, :, :]
        _, prob, _, point = cv2.minMaxLoc(probMap)
        x_p = (w * point[0]) / W + x
        y_p = (h * point[1]) / H + y
        if prob > 0.1:
            points.append((int(x_p), int(y_p)))
        else:
            points.append(None)
    points_all.append(points)
    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0,255,255), 2)
            cv2.circle(frame, points[partA], 4, (0,0,255), -1)

cv2.imwrite('result/yolo_openpose_result.jpg', frame)
print('Output saved to result/yolo_openpose_result.jpg')
