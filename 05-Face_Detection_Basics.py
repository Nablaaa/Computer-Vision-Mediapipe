import mediapipe as mp
import cv2
import time


cap = cv2.VideoCapture('Body-videos/2.mp4')
#cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75) # confidence


while True:
	success, img = cap.read()
	
	
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	results = faceDetection.process(imgRGB)
	
	if results.detections:
		for id, detection in enumerate(results.detections):
			#mpDraw.draw_detection(img, detection)	
		
		
			bboxC = detection.location_data.relative_bounding_box
			h, w, c = img.shape
			
			bbox = int(bboxC.xmin * w), int(bboxC.ymin * h),\
					int(bboxC.width * w), int(bboxC.height * h)
					
					
			cv2.rectangle(img, bbox,(255,0,255),2)
			cv2.putText(img, str(int(detection.score[0]*100)) + " %",
						(bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
						2, (255,200,0),3)
	
	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime
	
	cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,
				3, (255,0,0),3)
	
	cv2.imshow("Image",img)	
	
	
	
	cv2.waitKey(1)
	
