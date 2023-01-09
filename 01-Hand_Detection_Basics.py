# Projekt 1
# Hand landkarte

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

# input: 
# static_image_mode to detect/track = False/True
# max_num_hands
# min_detection_confidence
# min_tracking_confidence
hands = mpHands.Hands() # use default parameters

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# open camera
while True: 
    success, img = cap.read()
    
    # BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRGB)

    # uncomment to check, if hand is detected
    # print(results.multi_hand_landmarks) # none if no hand visible
    
    # detect multiple hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # index of finger landmarks and lm itself
                
                h, w, c = img.shape # width, height, channel of image
                
                # lm is a ratio of the display size
                # multiply with display height, width to get pixel
                # position
                cx, cy = int( lm.x * w), int(lm.y * h) 
                
                # print(id, cx,cy)
                
                # go for a single piont
                # e.g. id zero is the wrist
                # id 4 is the thumb tip
                if id == 4: 
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                if id == 8: 
                    cv2.circle(img, (cx,cy), 15, (100,0,255), cv2.FILLED)
                
            # draw 
            # hand landmarks = visible points
            # mpHands.HAND_CONNECTIONS = lines between points
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)



		
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    # display it on screen
    # input:
    # on which image
    # text is a string element of the integer "frames per second"
    # position
    # any font for the text
    # text size
    # text color in (BGR)
    # thickness of line
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 100, 0), 3)
    
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
    
    
