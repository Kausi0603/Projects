import cv2

#Trained data Set

trainedDatasset =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read image

img=cv2.imread('images/suriya.jpg')


#convert into grey scale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=trainedDatasset.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Display',img)
#cv2.imshow('Gray',gray)
cv2.waitKey()


