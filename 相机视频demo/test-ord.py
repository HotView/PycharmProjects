import cv2

a = ord("q")
b =ord("w")
cv2.namedWindow("video")
while True:
    b= cv2.waitKey(10)
    print("anjian :",b)


print(a,b)
print(ord('a'),ord('A'))