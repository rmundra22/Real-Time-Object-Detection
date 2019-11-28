# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def agr_parser():
	"""construct the argument parse and parse the arguments"""
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	return args

def initilizing_video_stream():
	"""initialize the video stream, allow the cammera sensor to warmup,
	   and initialize the FPS counter"""
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()
	return vs, fps

def preprocess_video_frames(vs):
	"""grab the frame from the threaded video stream and resize it
	   to have a maximum width of 400 pixels"""
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# convert frame into a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	return blob, frame

def serializing_model(args):
	"""load our serialized model from disk"""
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	return net

def get_detections_from_frames(blob, net):
	"""pass the blob through the network and obtain the detections and
	   predictions"""
	net.setInput(blob)
	detections = net.forward()
	return detections

def filter_out_detections(args, detections, confidence, key):
	"""filter out weak detections by ensuring the `confidence` is
	   greater than the minimum confidence"""
	if confidence > args["confidence"]:
		# extract the index of the class label from the
		# `detections`, then compute the (x, y)-coordinates of
		# the bounding box for the object
		idx = int(detections[0, 0, key, 1])
	else:
		idx = None
	return idx

def draw_bounding_box(detections, frame, key):
	"""Draw bounding boxes for the filtered detections"""
	# grab the frame dimensions
	(h, w) = frame.shape[:2]
	# get coordinates for bounding boxes
	box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
	return box

def predict_class_labels(confidence, idx):
	# Get labels for the detections
	label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
	return label

def draw_predictions_on_frames(box, label, frame, idx):
	(startX, startY, endX, endY) = box.astype("int")
	# draw the prediction on the frame
	cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(frame, label, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	return frame

def main():
	# Parsing the arguments for real time object detection
	args = agr_parser()

	# Serializing the pre-trained model
	net = serializing_model(args)

	# Initializing the video stream as frames (fps - frames per second)
	vs, fps = initilizing_video_stream()

	# loop over the frames from the video stream
	while True:
		# Preprossesing the frames to obtain detections
		blob, frame = preprocess_video_frames(vs)

		# Obtaining detections from each frame
		detections = get_detections_from_frames(blob, net)
		
		# loop over the detections
		for key in np.arange(0, detections.shape[2]):
			# extract the confidence associated with the prediction
			confidence = detections[0, 0, key, 2]

			# filter out weak detections
			idx = filter_out_detections(args, detections, confidence, key)

			if(idx == None):
				break
			else:
				# Get the bounding boxes for the detections
				box = draw_bounding_box(detections, frame, key)
				
				# Get predictions for the boxes
				label = predict_class_labels(confidence, idx)

				# Draw predictions on frames
				frame = draw_predictions_on_frames(box, label, frame, idx)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == '__main__':
	main()