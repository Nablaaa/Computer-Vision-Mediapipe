import cv2
import time
import numpy as np
import Hand_Tracking_Module as htm

##########################
wCam, hCam = 640, 480
minDist = 40
maxDist = 300
##########################



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0


# use htm with higher hand confidence
detector = htm.handDetector(detectionCon=0.7)


while True: 
	success, img = cap.read()
	
	detector.findHands(img)
	lmList = detector.findPosition(img, draw=False)
	
	if lmList:
		#print(lmList[4], lmList[8])
		
		x1, y1 = lmList[4][1], lmList[4][2]
		x2, y2 = lmList[8][1], lmList[8][2]
		
		
		cx, cy = (x1+x2)//2 , (y1+y2)//2 
		cv2.circle(img, (x1,y1), 15, (255,0,0),cv2.FILLED)
		cv2.circle(img, (x2,y2), 15, (255,0,0),cv2.FILLED)
		cv2.circle(img, ( cx , cy ), 15, (0,0,255),cv2.FILLED)
		cv2.line(img, (x1,y1), (x2,y2), (255,0,255),3)
		
		
		dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
		if dist < 50:
			cv2.circle(img, ( cx , cy ), 15, (0,255,0),cv2.FILLED)
		
		# rescale min max to 0, 100 percent		
		# min dist = 40
		# max dist = 300
		
		per = np.interp(dist, [minDist, maxDist], [0, 100])
		
		volBar = np.interp(dist, [40, 300], [400, 150])
		cv2.rectangle(img, (50, int(volBar)), (85,400), (255,0,0), cv2.FILLED)
		cv2.putText(img, str(int(per)) + " %", (40, 450), cv2.FONT_HERSHEY_PLAIN,
			    3, (255, 0, 0), 3)
			    
			    
	cv2.rectangle(img, (50, 150), (85,400), (255,0,0), 3)
	
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime

	# display it on screen
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
			    3, (255, 100, 0), 3)
	
	
	
	cv2.imshow("Image",img)
	cv2.waitKey(1)
		
