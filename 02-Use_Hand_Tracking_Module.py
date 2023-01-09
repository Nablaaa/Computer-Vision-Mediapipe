
import cv2
import mediapipe as mp
import time
import Hand_Tracking_Module as htm



	 
def main():
    
	pTime = 0
	cTime = 0
	cap = cv2.VideoCapture(0)

	detector = htm.handDetector()

    
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
	
	
	
	


    
    
    
