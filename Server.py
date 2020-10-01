#face detection dependencies
import cv2
import numpy as np

#TCP/IP socket communication dependencies
import socket
from threading import Thread
import threading
import time

#sys lib for checking a platform (OS)
from sys import platform

#support libraries for image packing
import struct
import pickle

#arguments parser for IP address enter
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='set the IP address.')
parser.add_argument('--IP', type=str, help='set the IP address of the rPi (server device)', default=socket.gethostname() )
parser.add_argument('--display', type=str2bool, help='set the display flag', nargs='?', const=True, default=False)
parser.add_argument('--fps', type=float, help='set the fps of rec video', default=10.0)
parser.add_argument('--streaming', type=int, help='set the streaming limit if hcsr04 detected someone', default=10)

args = parser.parse_args()

#RPi lib for distance measurement usecase
RPI_used = True
try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    RPI_used = False

DistanceDetection = True
if True == RPI_used:
    # GPIO Mode (BOARD / BCM)
    GPIO.setmode(GPIO.BCM)

    #set GPIO Pins
    GPIO_TRIGGER = 18
    GPIO_ECHO = 24

    #set GPIO direction (IN / OUT)
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)

    def distance():
        # set Trigger to HIGH
        GPIO.output(GPIO_TRIGGER, True)
    
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, False)
    
        StartTime = time.time()
        StopTime = time.time()
    
        # save StartTime
        while GPIO.input(GPIO_ECHO) == 0:
            StartTime = time.time()
    
        # save time of arrival
        while GPIO.input(GPIO_ECHO) == 1:
            StopTime = time.time()
    
        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2
    
        return distance

    def HCSR04_loop():
        global detected
        detected = False
        itterations = 0 #til itterationLimit
        itterationLimit = args.streaming
        distanceLimit = 80.0
        while DistanceDetection:
            dist = distance()
            print ("Measured Distance = %.1f cm" % dist)
            if(dist < distanceLimit):
                detected = True
            if(True == detected):
                itterations += 1
            if(itterations == itterationLimit):
                itterations = 0
                detected = False
            print("HCSR04_loop thread :: detected movement = ",str(detected) )
            time.sleep(1)
    
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


#face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#WebCam handler
if platform == "linux" or platform == "linux2":
    cap = cv2.VideoCapture(0)
elif platform == "win32":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 320)
cap.set(4, 240)

#socket server's IP address & port 
port = 21000
host = str(args.IP) #'192.168.0.109' #socket.gethostname() # Get local machine name

#socket initialisation
clients = set()
clients_lock = threading.Lock()

serversock = socket.socket()
serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversock.bind((host,port))
serversock.listen()

print("IP: ")
print(serversock.getsockname() )

th = []

#initialisation for data packing
ret, frame = cap.read()
result, frame = cv2.imencode('.jpg', frame, encode_param)
data = pickle.dumps(frame, 0)

ClientConnection = True

def listener(client, address):
    global data
    global detected
    # Atribut global govori da je u pitanju globalna promenljiva, te da ne instancira
    # novu lokalnu promenljivu nego njene vrednosti očitava ‘spolja’
    global sndMsg
    sndMsg = False
    print ("\nAccepted connection from: ", address,'\n')
    with clients_lock:
        clients.add(client)

    while ClientConnection:
        if detected == True:
            if(sndMsg == True):
                try:
                    client.sendall(struct.pack(">L", len(data) ) + data)
                    sndMsg = False
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
    
    print("\nBroken connection from: ", address, "\n")
    clients.remove(client)

def clientReceivement():
    print ("\nWaiting for new clients...\n")
    while True:
        try:
           (client, address) = serversock.accept()
        except OSError:
            break

        th.append(Thread(target=listener, args = (client,address)) )
        th[-1].start()

RecordVideo = True

from datetime import datetime

import os
mypath = 'videos/'
if not os.path.isdir(mypath):
   os.makedirs(mypath)

def VideoWriting():
    global frame
    global writeVideo
    global detected
    detected = False
    firstTime = True
    writeVideo = False
    out = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    while RecordVideo:
        if(True == firstTime and True == detected and True == writeVideo):
            fileName = mypath +'video_' + str(datetime.now() ) + '.avi'
            out = cv2.VideoWriter(fileName, fourcc, args.fps, (int(cap.get(3)), int(cap.get(4)))) 
            firstTime = False
        if(True == writeVideo):
            out.write(frame)
            writeVideo = False
        if(False == detected):
            firstTime = True
            out.release()
    if(out.isOpened() ):
        out.release()

if True == RPI_used:
    th.append(Thread(target=HCSR04_loop))
    th[-1].start()

th.append(Thread(target=clientReceivement) )
th[-1].start()

th.append(Thread(target=VideoWriting) )
th[-1].daemon = True
th[-1].start()

incr = 0
limit = 40000

detected = not RPI_used
sndMsg = False
writeVideo = False
while True:
    try:        
        incr = incr + 1

        if(incr < limit):
            continue
        elif(incr == limit):
            incr = 0

    except KeyboardInterrupt:
        break

    try:
        ret, frame = cap.read()

        if(ret == False):
            break

        if detected == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 
                scaleFactor=1.3, 
                minNeighbors=5)
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
            
            # #enable only sending frames with face detection
            # if(len(faces) == 0):
            #     continue
            
            sndMsg = True
            writeVideo = True
            result, framePacked = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(framePacked, 0)

        if(bool(args.display) == True):
            cv2.imshow('img', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                detected = not detected

    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()

for client in clients:
    client.shutdown(socket.SHUT_RDWR)
    client.close()

ClientConnection = False
DistanceDetection = False
RecordVideo = False

try:
    serversock.shutdown(socket.SHUT_RDWR)
    serversock.close()
except OSError:
    serversock.close()

if(clients_lock.locked() == True):
    clients_lock.release()

for thd in th:
    thd.join()

print("\nSuccessfully closed server application\n")
exit() 