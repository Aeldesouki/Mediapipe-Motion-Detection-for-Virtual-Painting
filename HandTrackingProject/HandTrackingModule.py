import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4,8,12,16,20] #Stores the landmark points for the tips of each finger

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon, )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumber = 0, draw = True):
        xList = []
        yList = []
        bBox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            if handNumber < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNumber]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if xList and yList:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bBox = xmin, ymin, xmax, ymax
                    if draw:
                        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

                return self.lmList, bBox

        # If no hand detected or xList is empty, return nothing
        return None

    def checkFingers(self):
        fingersUp = []

        #Responsible for thumbs
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingersUp.append(1)
        else:
            fingersUp.append(0)

        #Responsible for fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingersUp.append(1)
            else:
                fingersUp.append(0)



        return fingersUp



def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])

        currentTime = time.time()
        framesPerSecond = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(framesPerSecond)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()