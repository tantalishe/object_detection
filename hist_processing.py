import cv2
import numpy as np

def get_hist_image(hist):

    # parameters of histogram image
    thickness = 2
    height = 310
    width = 256 * thickness

    image_hist = np.zeros((height,width,3), np.uint8)

    # normalize
    hist_norm = (hist.astype(float) / hist.max() * (height - 10)).astype(int)

    # drawing
    for index in range(256):
        cv2.rectangle(image_hist,(index * thickness, height - hist_norm[index]),((index + 1) * thickness - 1, height),(0,255,0))
        # cv2.rectangle(image_hist,(index * thickness, 0),(index * thickness + thickness - 1, hist_norm[index]),(0,255,0))
        # cv2.circle(image_hist,(0,0), 10, (0,0,255), -1) #red
        # cv2.circle(image_hist,(width,height), 10, (0,255,0), -1) #green
        # cv2.circle(image_hist,(width,0), 10, (255,255,255), -1) #white
        # cv2.circle(image_hist,(0,height), 10, (255,0,0), -1) #blue
    return image_hist