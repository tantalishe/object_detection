import cv2
import numpy as np


if __name__ == '__main__':
    cam = cv2.VideoCapture(1)
    image = None
    while True:
        _, frame = cam.read()
        image = frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
        
    scale = 1
    # image = cv2.imread('circle.png')
    w, h, _ = image.shape
    image = cv2.resize(image, (int(h*scale), int(w*scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (7, 7))
    # gray = cv2.bitwise_not(image)
    ret, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # th = cv2.adaptiveThreshold(gray, 255, 1, cv2.THRESH_BINARY, self.blocksize, self.c)

    v = np.median(image)
    sigma = 0.33
    canny_low = int(max(0, (1 - sigma) * v))
    canny_high = int(min(255, (1 + sigma) * v))
    edged = cv2.Canny(th, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=3)
    edged = cv2.erode(edged, None, iterations=2)

    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0))
    
    print(len(contours))
    p = cv2.arcLength(contours[0], True)
    c = cv2.approxPolyDP(contours[0], 0.001 * p, True)
    
    cv2.drawContours(image, c, -1, (0, 0, 255), 3)
    
    np.savez('squar50', c)
    
    # with np.load('squar50.npz') as X:
    #     c = [X[i] for i in X]
    # cv2.drawContours(image, c, -1, (0, 0, 255), 3)


    cv2.imshow('lol1', th)
    cv2.imshow('lol', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
