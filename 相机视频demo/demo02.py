import cv2

capture = cv2.VideoCapture(0)

success,frame = capture.read()
print(frame.shape)

cv2.VideoWriter_fourcc('I','4','2','0')
cv2.VideoWriter_fourcc('P','I','M','1')
cv2.VideoWriter_fourcc('X','V','I','D')
cv2.VideoWriter_fourcc('T','H','E','O')
cv2.VideoWriter_fourcc('F','L','V','1')