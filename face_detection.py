# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function
# def resize(img, size) :
#     width = int(img.shape[1]*size)
#     height = int(img.shape[0] * size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
first = face_rec.load_image_file('images/dimashi.jpg')
first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
# first = resize(first, 0.50)
second = face_rec.load_image_file('images/dimashi.jpg')
# second = resize(second, 0.50)
second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)

# # finding face location
img_encoding = face_rec.face_encodings(first)[0]
img_encoding2 = face_rec.face_encodings(second)[0]

result = face_rec.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.putText(second, f'{result}', (10,30), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2 )

cv2.imshow('main_img', first)
cv2.imshow('test_img', second)
cv2.waitKey(0)
cv2.destroyAllWindows()