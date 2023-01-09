import cv2
import time
import numpy as np
import Hand_Tracking_Module as htm
import os

##########################
#wCam, hCam = 640, 480
wCam, hCam = 1080, 720
menu_height = 90
draw_dist = 80
eraserThickness = 20
##########################



folder = "header"
myList = os.listdir(folder)
print(myList)
overlayList = []
for imPath in myList:
	image = cv2.imread(f'{folder}/{imPath}')	
	overlayList.append(image)
	
header = overlayList[4]
print(header.shape)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0


# use htm with higher hand confidence
detector = htm.handDetector(detectionCon=0.9,maxHands=1)

color = (0,0,0)
xp, yp = 0, 0  

imgCanvas = np.zeros((hCam, 200+wCam,3), np.uint8)

pos = 570 # mark the color
while True: 
	success, img = cap.read()
	img = cv2.flip(img, 1)
	img[0:80, 0:640] = header	
	
	detector.findHands(img)
	lmList = detector.findPosition(img, draw=False)
	
	
	
	if lmList:
		#print( lmList[8])

		# get landmarks		
		x1, y1 = lmList[4][1], lmList[4][2] # 1 st thumb
		x2, y2 = lmList[8][1], lmList[8][2] # 2 nd
		
		
		
		brushThickness = 6
		
		# red		120
		# yellow	240
		# blue		380
		# green		500
		# eraser
		
		# go to menu
		if y2 < menu_height:
			
			if x2 > 0 and x2 < 120:
				color = (0,0,255)
				pos = 50
				
			elif x2 > 120 and x2 < 240:
				color = (0,255,255)
				pos = 190
				
			elif x2 > 240 and x2 < 380:
				color = (255,0,0)
				pos = 320
				
			elif x2 > 380 and x2 < 500:
				color = (0,255,0)
				pos = 450
				
			elif x2 > 500 :
				color = (0,0,0)
				brushThickness = eraserThickness
				pos = 570
		
		
		dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
		
		cv2.circle(img, ( x2 , y2 ), 3 * brushThickness, color,cv2.FILLED)
		
		
		if dist < draw_dist:
			cv2.circle(img, ( x2 , y2 ), brushThickness, color,cv2.FILLED)		
			
			# drawing
			cv2.line(imgCanvas, (xp,yp), (x2,y2), color, brushThickness)
		
		xp, yp = x2, y2   
	
	cv2.rectangle(img, (pos - 20, 80), (pos + 20,100), color, cv2.FILLED)
		
		
	imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
	_, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
	imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

	#imgOverlay = cv2.bitwise_and(img, imgInv)
	imgOverlay = cv2.bitwise_or(img, imgCanvas)
	
		
	#cv2.imshow("Image",img)
	cv2.imshow("Canvas",imgCanvas)
	cv2.imshow("Overlay", imgOverlay)
	cv2.waitKey(1)
		
		
		
		
		
		
		
		
		
		
		
