# Projekt 1
# Hand landkarte

import cv2
import mediapipe as mp
import time




class handDetector():
	def __init__(self, mode=False, maxHands=2,
					complexity=1, detectionCon=0.5, 
					trackCon=0.5):
		# create object
		self.mode = mode
		self.maxHands = maxHands
		self.complexity = complexity
		self.detectionCon = detectionCon
		self.trackCon = trackCon
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, 
										self.maxHands,
										self.complexity,
										self.detectionCon,
										self.trackCon) 
		self.mpDraw = mp.solutions.drawing_utils
		
		
	def findHands(self, img, draw=True):
        # BGR to RGB
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
		self.results = self.hands.process(imgRGB)


		# detect multiple han
		
		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				for id, lm in enumerate(handLms.landmark):
					if draw:
						self.mpDraw.draw_landmarks(img,handLms, 
										self.mpHands.HAND_CONNECTIONS)
	  
		return img 
	  	
	  	
	def findPosition(self, img, handNo=0, draw=True):
		
		lmList = []
		
		# if hands available
		if self.results.multi_hand_landmarks:
		
			# do the concept only for a single hand (more is possible)
			myHand = self.results.multi_hand_landmarks[handNo]
				
			for id, lm in enumerate(myHand.landmark):
				# index of finger landmarks and lm itself
				h, w, c = img.shape # width, height, channel of image
				
				
				# lm is a ratio of the display size
				# multiply with display height, width to get pixel
				# position
				cx, cy = int( lm.x * w), int(lm.y * h) 

				#print(id, cx,cy)
				lmList.append([id, cx,cy])
				
				if draw:
					# go for a single piont
					# e.g. id zero is the wrist
					# id 4 is the thumb tip
					if id == 4:
					  cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
					if id == 8: 
						cv2.circle(img, (cx,cy), 15, (100,0,255), cv2.FILLED)
				
		return lmList	
	
	  				
	  
	  
	  
		




	 
def main():
    
	pTime = 0
	cTime = 0
	cap = cv2.VideoCapture(0)

	detector = handDetector()

    
	# open camera
	while True: 
		success, img = cap.read()
		
		# now give the image to the find hands 
		img = detector.findHands(img)
		
		lmList = detector.findPosition(img)
		
		if len(lmList)!=0:
			# any index
			print(lmList[4])
		
		
		cTime = time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime

		# display it on screen
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
				    3, (255, 100, 0), 3)
		
		cv2.imshow("Image",img)
		cv2.waitKey(1)
if __name__ == "__main__":
	main()
	
	
	
	


    
    
    
