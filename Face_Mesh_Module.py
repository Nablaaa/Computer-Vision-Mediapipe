import cv2
import mediapipe as mp
import time 


class FaceMeshDetector():

	def __init__(self,staticMode = False, maxFaces=2, refine_lm = True,
					minDetectionCon=0.5, minTrackCon=0.5,
					thickness = 2, circle_radius=1):
		
		self.staticMode =staticMode
		self.maxFaces =maxFaces
		self.refine_lm = refine_lm
		self.minDetectionCon =minDetectionCon
		self.minTrackCon =	minTrackCon	
		
		self.thickness = thickness	
		self.circle_radius = circle_radius
					
		self.mpDraw = mp.solutions.drawing_utils
		self.mpFaceMesh = mp.solutions.face_mesh
		self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
										self.maxFaces,
										self.refine_lm,
										self.minDetectionCon,
										self.minTrackCon)
		self.drawSpec = self.mpDraw.DrawingSpec(self.thickness, 
								self.circle_radius)



	def findFaceMesh(self, img, draw=True):
		self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.faceMesh.process(self.imgRGB)
		
		faces = []
		if self.results.multi_face_landmarks:
					
			for faceLms in self.results.multi_face_landmarks:
				
				face = []
				for idx, lm in enumerate(faceLms.landmark):
					h, w, c = img.shape
					x,y = int(lm.x *w), int(lm.y*h)
					face.append([x,y])
					
				faces.append(face)	
				
				if draw:
					self.mpDraw.draw_landmarks(img, faceLms,
							self.mpFaceMesh.FACEMESH_CONTOURS,
							self.drawSpec,self.drawSpec)
		return img, faces
		
					
		
	
def main():
	#cap = cv2.VideoCapture('Body-videos/2.mp4')
	cap = cv2.VideoCapture(0)

	
	pTime = 0

	detector = FaceMeshDetector()

	while True:
		success, img = cap.read()
		img, faces = detector.findFaceMesh(img)
		
		if faces:
			print(len(faces))
		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime
		
		cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,
					3, (255,0,0),3)
					
					
		cv2.imshow("Image",img)	
		cv2.waitKey(20)

if __name__ == "__main__":
	main()
	
	





