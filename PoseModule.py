import cv2
import mediapipe as mp
import time
import math





class poseDetector():
	def __init__(self, mode=False, upBody=False,smooth=True,
					detectionCon=0.7, trackConf=0.7 ):
		

		self.mode = mode
		self.upBody = upBody
		self.smooth = smooth
		self.detectionCon = detectionCon
		self.trackConf = trackConf

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode,1, self.upBody, 
									self.smooth, self.detectionCon,
									self.trackConf)
		

	def findPose(self, img, draw = True):
		
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.pose.process(imgRGB)
				
		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
									self.mpPose.POSE_CONNECTIONS)
	
		return img
	
	
	
	def findPosition(self, img, draw=True):
		self.lmList = []
		
		if self.results.pose_landmarks:
	
			for id, lm in enumerate(self.results.pose_landmarks.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				self.lmList.append([id, cx, cy])	
				
				if draw:
					cv2.circle(img, (cx,cy),5,(255,0,0),cv2.FILLED)
		return self.lmList

	def findAngle(self, img, p1, p2, p3, draw=True):
		# lamdmarks		
		_, x1, y1 = self.lmList[p1]
		_, x2, y2 = self.lmList[p2]
		_, x3, y3 = self.lmList[p3]
		
		
		# angle
		angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
								math.atan2(y1-y2, x1-x2))
		
		if draw:
		
			cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
			cv2.line(img, (x2,y2), (x3,y3),  (255,255,255), 3)
		
			cv2.circle(img, (x1,y1), 5, (0,0,255),cv2.FILLED)
			cv2.circle(img, (x2,y2), 5, (0,0,255),cv2.FILLED)
			cv2.circle(img, (x3,y3), 5, (0,0,255),cv2.FILLED)
			
			cv2.circle(img, (x1,y1), 15, (0,0,255),2)
			cv2.circle(img, (x2,y2), 15, (0,0,255),2)
			cv2.circle(img, (x3,y3), 15, (0,0,255),2)
			
			
			
			
			
		return angle
	
def main():

	cap = cv2.VideoCapture('Body-videos/1.mp4')
	#cap = cv2.VideoCapture(0)
	pTime = 0

	detector = poseDetector()
	
	while True:
		success, img = cap.read()
		img = detector.findPose(img)
		lmList= detector.findPosition(img)
		
		if lmList:
		
			# highlight a special index
			cv2.circle(img, (lmList[14][1],lmList[14][2]),15,(255,255,0),cv2.FILLED)
	
		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime
		
		cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,
					3, (255,0,0),3)
		
		cv2.imshow("Image", img)
		cv2.waitKey(1) #1ms delay
	
	

	
if __name__ == "__main__":
	main()
	
	
	
	
	
	
	
	
	
	
	
