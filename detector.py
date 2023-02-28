import cv2
import numpy as np


cap = cv2.VideoCapture("cars_moving.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100)  #since the camera is stable, we can increase the history

#The logic behind history is, how past the computer comparing the current frame with.
#If the camera was also moving, if we set a long history, since the camera is moving even the objects that are not
#moving would be detected. But in this case since the camera is stable, we can compare with longer history.

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use the MP4V codec
output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    
    ret,frame = cap.read()
    

    #Moving Object Detection
    mask = object_detector.apply(frame)

    #We are going to apply threshold because we are also detecting the shadow of vehicles.
    #Shadows are closer to the color gray so we will use binary

    _,mask_threshold = cv2.threshold(mask,250,255,cv2.THRESH_BINARY) #any pixel below whiteish will turn black

    #Clearing small white dots
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(mask,kernel)

    #Finding the contours on the erosion
    contours,_ = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Drawing contours onto original video
    for contour in contours:
        #Only drawing the contours that has an area bigger than 100
        #so that we can get rid of other small moving objects and noises
        if cv2.contourArea(contour) < 3000:
            continue
        else:
            #cv2.drawContours(frame,[contour],0,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)

    cv2.imshow("Video",frame)

    output_video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows