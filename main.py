import numpy as np
import cv2
import math

def rotateImage(image, angle, scale):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=1)
  return result


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

penis = cv2.imread("penis.png", cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture(0)



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    ret, img = cap.read()
    height, width, channels = img.shape
    penis = cv2.resize(penis, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            #for (ex,ey,ew,eh) in eyes:
                # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                # cv2.circle(roi_color, (int(ex+ew/2), int(ey+eh/2)), int(ew/2), (147,20, 255), 5, 1)
            a = (eyes[1][1]-eyes[1][0])/(eyes[1][0]-eyes[0][0])
            dist = calculateDistance(eyes[0][0], eyes[1][0], eyes[0][1], eyes[1][1])
            scale = dist/width
            angle = math.atan2(eyes[1][1]-eyes[1][0],eyes[1][0]-eyes[0][0])
            eyesCenter = ((eyes[1][0]+eyes[0][0])/2, (eyes[1][1]+eyes[0][1])/2)
            #print(eyes)
            print(eyesCenter)
            offset = (eyesCenter[0]+x-width/2+30, eyesCenter[1]+80+y-height/2)
            #print(offset)
            #print(offset)
            print(angle)
            rot = rotateImage(penis, angle+180, 2*scale+0.2)
            #rot = penis
            scal=rot
            trans = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
            print(trans)
            przes = cv2.warpAffine(scal, trans, scal.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=1)
            cv2.imshow('penis', przes)
            alfa = przes[:, :, 3]/255.0
    for c in range(0, 3):
        img[alfa>0]= [147, 20, 255]

    img = cv2.resize(img, (1280, 960))
    cv2.imshow('img',img)
    cv2.waitKey(1)
cv2.destroyAllWindows()