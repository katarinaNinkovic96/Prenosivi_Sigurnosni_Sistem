import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

from threading import Thread
import threading
import time

# (grabbed, frame) = cap.read()
# fshape = frame.shape
# fheight = fshape[0]
# fwidth = fshape[1]
# print (fwidth , fheight)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (fwidth,fheight))

recordVideo = True
detected = True

from datetime import datetime

def VideoWriting():
    global frame
    global writeVideo
    firstTime = True
    writeVideo = False
    out = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #cv2.VideoWriter_fourcc(*'XVID')

    while recordVideo:
        if(True == firstTime and True == detected and True == writeVideo):
            print("init")
            print(frame.shape[0], frame.shape[1])
            fileName = str(datetime.now() ) + '.avi'
            out = cv2.VideoWriter(fileName, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))) 
            #cv2.VideoWriter('output.avi',fourcc, 20.0, (frame.shape[0],frame.shape[1]) )
            print(out.getBackendName())
            firstTime = False
        if(True == writeVideo):
            out.write(frame)
            print("VideoWriting :: Frame")
            writeVideo = False
        if(False == detected):
            firstTime = True
            out.release()
    
    print("Out")
    if(out.isOpened() ):
        print("Released")
        out.release()

th = []

th.append(Thread(target=VideoWriting) )
th[-1].daemon = True
th[-1].start()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        # out.write(frame)

        writeVideo = True
        print("Main :: Frame")
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) 
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            detected = not detected
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

recordVideo = False

for thd in th:
    thd.join()