from pytesseract import Output
import pytesseract
import datetime
import imutils
import time
import cv2
import os
import sys
import numpy as np
from PIL import Image
from imutils import contours

recognizer = cv2.face_LBPHFaceRecognizer.create()
path='./dataSet'

def getImgensAndTrainingRecognizerFaces(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagesPaths in imagePaths:
        faceImg=Image.open(imagesPaths,mode='r').convert('L');
        faceNP=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagesPaths)[-1].split('.')[1])
        faces.append(faceNP)
        print ("treinando o reconhecimento do rosto do ID: " + str(ID))
        IDs.append(ID)
        cv2.imshow("treinando" , faceNP)
        cv2.waitKey(10)
    recognizer.train(faces,np.array(IDs))
    recognizer.save("trainingData.json")
    cv2.destroyAllWindows()
    return


def getImagensToTraining():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    cam=cv2.VideoCapture(0);

    id=input('Informe o user ID: ')
    sampleNum=0;
    while (True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h)in faces:
            sampleNum=sampleNum+1;
            cv2.imwrite("./dataSet/USer."+str(id)+"."+str(sampleNum)+ ".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Face",img);
        if(sampleNum>=20):
            break;
        if(cv2.waitKey(1)==ord('q')): 
            break;                               
    cam.release()
    sampleNum=0;
    img=None;
    cv2.destroyAllWindows();
    return;

def recognizerFaceOnCam():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    cam=cv2.VideoCapture(0);
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("./trainingData.json")
    id=0;
    font = cv2.FONT_ITALIC 
    while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])
            if(id == 1):
                id="Thomaz" 
            if(id == 2):
                id="pessoa 2" 
            if (id == 3):
                id="Pessoa 3" 
            if(id == 4):
                id="Pessoa 4"
            if(id == 5):
                id="Pessoa 5"
            cv2.putText(img,str(id),(x+w,y), font, 1,(77, 6, 70),1,cv2.LINE_AA)
        cv2.imshow("Face",img);
        if(cv2.waitKey(1)==ord('q')): 
            break;                               
    cam.release()
    cv2.destroyAllWindows();
    return;

def motionDetection():
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)

    firstFrame = None

    
    while True:

        (grabbed, frame) = camera.read()
        text = "Sem Movimento"

        if not grabbed:
            break

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue


        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]



        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)


        for c in cnts:

            if cv2.contourArea(c) > 2500:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Movimento Detectado!"


        cv2.putText(frame, "Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p" + " -- To exit press ""q"""),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    return;

def recVideo():
    # Create an object to read
    # from camera
    video = cv2.VideoCapture(0)

    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.mp4' file.
    result = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, size)

    while(True):
        ret, frame = video.read()

        if ret == True:

            cv2.putText(frame, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            (10, frame.shape[0] - 460), cv2.FONT_ITALIC, 0.35, (0, 0, 255), 1)
            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)

            # Display the frame
            # saved in the file
            cv2.imshow('Frame', frame)

            # Press S on keyboard
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Break the loop
        else:
            break

    # When everything done, release
    # the video capture and video
    # write objects
    video.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")


    return;

def ORCDetection():
    video = cv2.VideoCapture(0)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to localize each area of text in the input image
    # Use the attached camera to capture images
    # 0 stands for the first one
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(Image.fromarray(img1))
        cv2.imshow('frame', img1)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        print("Extracted Text: ", text)
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return;


def menu(path):                   
    choice ='0'
    while (True):
        print("Usado OpenCV: 4.3.0 , Python 3.8.4 ")
        print("Escolha uma das Opções abaixo: ")
        print("1 - Para Capturar um novo rosto.")
        print("2 - Para Treinar o reconhecimento dos Rostos capturados.")
        print("3 - Para Reconhecer os rostos capturados.")
        print("4 - Detecção de movimentos.")
        print("5 - Gravar video da camera")
        print("6 - Reconhecimento de Textos")
        print("7 - Sair")

        

        choice = input ("Escolha uma opção: ")

        if choice == "1":
            getImagensToTraining()
        elif choice == "2":
            getImgensAndTrainingRecognizerFaces(path)
        elif choice == "3":
            recognizerFaceOnCam()
        elif choice == "4":
            motionDetection()
        elif choice == "5":
            recVideo()
        elif choice == "6":
            ORCDetection()
        elif choice == "7":
            cv2.destroyAllWindows()
            exit()
        else:
            print("Por favor selecione uma opção valida.")
menu(path);

