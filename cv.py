import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lo_gr = np.array([35, 40, 40])      
    up_gr = np.array([85, 255, 255])    

    mask = cv2.inRange(hsv, lo_gr, up_gr)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("masked", masked_img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
