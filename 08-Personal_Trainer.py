import cv2
import time
import numpy as np
import PoseModule as pm


##########################
wCam, hCam = 640, 480


minAngle= 105
maxAngle = 160
##########################


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Body-videos/5.mp4')
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0


# use htm with higher hand confidence
detector = pm.poseDetector(detectionCon=0.7)

down = 0
up = 0
counter = 0

while True: 
	success, img = cap.read()
	
	detector.findPose(img, draw=False)
	lmList = detector.findPosition(img, draw=False)
	
	
	
	x = []
	y = []
	
	
	cv2.rectangle(img, (20, 30), (90,80), (255,255,255), cv2.FILLED)	
	
	
	cv2.rectangle(img, (45, 410), (140,450), (255,255,255), cv2.FILLED)
	
	
	if lmList:
		
		angle = detector.findAngle(img,12,14,16,True)


		# rescale min max to 0, 100 percent		
		# min dist = 40
		# max dist = 300
		
		per = np.interp(angle, [minAngle, maxAngle], [400, 150])
		
		
		

		cv2.rectangle(img, (50, int(per)), (85,400), (255,0,0), cv2.FILLED)
		cv2.putText(img, str(int(angle)), (40, 450), cv2.FONT_HERSHEY_PLAIN,
			    3, (255, 0, 0), 3)
			    
			    
		if angle < minAngle and up == down:
			down = down + 1
			
		if angle > maxAngle and up < down:
			up = up + 1
			
		
		if up == down:
			counter = up
			
		cv2.putText(img, str(int(counter)), (40, 70), cv2.FONT_HERSHEY_PLAIN,
			    3, (255, 0, 0), 3)
		
			    
	cv2.rectangle(img, (50, 150), (85,400), (255,0,0), 3)
	
		
		
		
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime

	# display it on screen
	#cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
	#		    3, (255, 100, 0), 3)
	
	
	
	cv2.imshow("Image",img)
	cv2.waitKey(1)
		



