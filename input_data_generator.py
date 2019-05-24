import numpy as np
import argparse
import cv2
import os
import imutils
import glob
import time
from imutils.video import FPS


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="provide path to input video")
#ap.add_argument("-m", "--model", type = str, required = True, help="provide tensorflow .pb file")
#ap.add_argument("-p", "--pbtxt", type = str, required = True, help="provide tensorflow .pbtxt config file")
ap.add_argument("-o", "--output", type=str, required=True, help="provide path to store outputs")
ap.add_argument("-s", "--skip_frames", type=int, help="number of frames you want to be skipped", default = 12)
#ap.add_argument("-c", "--confidence", type=float, help="minimum confidence level to filter out weak detections", default = .50)
args = vars(ap.parse_args())

confidence = .50
project_dir = os.getcwd()
print(project_dir)
#providing frozen inference graph model and configuration file to detect faces
model = os.path.join(project_dir, "./face_detector/opencv_face_detector_uint8.pb")
configuration = os.path.join(project_dir, "./face_detector/face_detector.pbtxt")
#net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])
net = cv2.dnn.readNetFromTensorflow(model, configuration)

#making pointer to capture from the video file
#vs = cv2.VideoCapture(os.path.join(project_dir, args["input"]))
vs = cv2.VideoCapture(args["input"])
time.sleep(2.0)

#variable to keep track the number and sequence of images that are saved
image_no = 0
#initializing width and height of the frame
width = 0
height = 0

#to keep track of frames total frames will be counted
total_frames = 0
image_no = 472

#start the fps meter
fps = FPS().start()

while (True):
	(flag, frame) = vs.read()
	#frame = frame[1]	
	
	if not flag:
		print("Please provide an input video")
		break
	
	#refraining detector from detecting on each frame
	if total_frames % args["skip_frames"] == 0:
	#resizing frame for the sake of fast processing. have to keep in mind till training phase
	#frame = imutils.resize(frame, width=700)

		if height == 0 or width == 0:
			(height, width) = frame.shape[:2]
			print(str(height) + ":" + str(width))
		#float_image = cv2.UMat(frame)
		#blob_image = cv2.resize(float_image, 300, 300)
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 117.0, 123.0], swapRB=False)
		net.setInput(blob)
		detections = net.forward()
		if len(detections) > 0:
			for i in range(detections.shape[2]):
				credence = detections[0, 0, i, 2]
				if credence < confidence:
					continue
				detection_index = int (detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])

				(startX, startY, endX, endY) = box.astype("int")

				face_roi = frame[startY:endY, startX:endX]

				#writer = os.path.sep.join([args["output"], "{}.png".format(image_no)])
				writer = os.path.join(project_dir, args["output"], "{}.bmp".format(image_no))
				image_no += 1
				cv2.imwrite(writer, face_roi)
				#cv2.imshow('Face', face_roi)
				print("[Note] saved {} to disk".format(writer))
				# key = cv2.waitKey(1) & 0xFF
				# if key == ord('q'):
				# 	break
	total_frames += 1
	fps.update()
fps.stop()
print("[Note] elapsed time in seconds: {:.2f}".format(fps.elapsed()))
print("[Note] approximate FPS: {:.2f}".format(fps.fps()))
if not args.get("input", False):
	vs.stop()
else:
	vs.release()
# if writer is not None:
# 	writer.stop()
cv2.destroyAllWindows()