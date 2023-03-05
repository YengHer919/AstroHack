# importing libraries
import cv2
import numpy
import time

# Defining a function motionDetection   
def motionDetection():
    # capturing video in real time
    cap = cv2.VideoCapture(0)


    # reading frames sequentially
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2()
    count = 0
    isCountedAlready = False

    while cap.isOpened():
        #ret, frame = cap.read()
       # height, width, _ = frame.shape
         # Extract Region of interest
        roi = frame1[50: 200,300: 400]
        roi2 = frame2[50: 200,300: 400]

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        # difference between the frames
        diff = cv2.absdiff(roi, roi2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            isCountedAlready = False
        
        for contour in contours:
        
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.putText(frame1, "STATUS: {}".format('MOTION DETECTED'), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                     #   1, (217, 10, 10), 2)
            if not isCountedAlready:
                count = count + 1

            #else:
            #   return count

        # cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        cv2.putText(frame1, "Score: {}".format(str(count)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            1, (217, 10, 10), 2)
        cv2.imshow("Video", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
       # cv2.imshow("Video1", roi)

        if cv2.waitKey(50) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motionDetection()
