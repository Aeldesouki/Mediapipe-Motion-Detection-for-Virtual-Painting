import cv2
import numpy as np
import time
import os
import HandTrackingModule as ht


xPrevious, yPrevious = 0,0

paintThickness = 15
eraserThickness = 50

folderPth = "PaintImages"
ovrList = []
paintList = os.listdir(folderPth)
detection = ht.handDetector(detectionCon=0.90)
drawColor = (0, 0, 255) #red, in BGR....

for images in paintList:
    image = cv2.imread(f'{folderPth}/{images}')
    ovrList.append(image)

header = ovrList[0]


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

#creating a canvas to draw on
artCanvas = np.zeros((720,1280,3), np.uint8)



while True:
    #Importing image
    success, img = cap.read()
    #flip the image, for easier drawing (as the webcam shows the reflection)
    img = cv2.flip(img, 1)

    # Finding landmarks responsible for painting on the screen
    img = detection.findHands(img)
    result = detection.findPosition(img, draw=False)

    if result is not None:
        landmarkList, bBox = result

        #Making sure we have enough landmarks before accessing index 8 and 12
        if len(landmarkList) > 12:
            xIndex, yIndex = landmarkList[8][1:]
            xMiddle, yMiddle = landmarkList[12][1:]

            #Checking finger gestures to determine between selecting and painting
            checkFingersUp = detection.checkFingers()

            #Selection mode (two fingers)
            if checkFingersUp[1] and checkFingersUp[2]:
                xPrevious , yPrevious = 0,0
                print("Selection Mode")

                #check if we are at the top of the image
                if yIndex < 125: #Note:, the image is 125 px in length
                    if 250 < xIndex < 450: #Checking for a click
                        header = ovrList[0]
                        drawColor = (0, 0, 255) #red

                    elif 550 < xIndex < 750:
                        header = ovrList[1]
                        drawColor = (255, 0, 0) #blue

                    elif 800 < xIndex < 950:
                        header = ovrList[2]
                        drawColor = (0, 255, 0) #green

                    elif 1050 < xIndex < 1200:
                        header = ovrList[3]
                        drawColor = (0, 0, 0) #black

                cv2.rectangle(img, (xIndex, yIndex - 30), (xMiddle, yMiddle + 30), drawColor, cv2.FILLED)



            #If drawing, (index finger)
            if checkFingersUp[1] and checkFingersUp[2] == False:
                cv2.circle(img, (xIndex, yIndex), 10, drawColor, cv2.FILLED)
                print("Drawing Mode")

                #Checking if the painter has JUST started, simply set the previous points to the current points
                if xPrevious == 0 and yPrevious == 0:
                    xPrevious, yPrevious = xIndex, yIndex

                #special condition for erasing, mainly paint thickness is greater for easier erasing
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, eraserThickness)
                    cv2.line(artCanvas, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, paintThickness)
                    cv2.line(artCanvas, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, paintThickness)


                xPrevious, yPrevious = xIndex, yIndex




    imgGray = cv2.cvtColor(artCanvas, cv2.COLOR_BGR2GRAY)
    _, imageInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imageInv = cv2.cvtColor(imageInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imageInv)
    img = cv2.bitwise_or(img, artCanvas)

    #Lines of code responsible for setting the paint images on the top of the webcam.
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, artCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", artCanvas)
    cv2.waitKey(1)


