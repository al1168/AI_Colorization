import cv2

import cv2
import numpy as np

#greying and scalling hte shape of the image
def scale(src):
    src = src.tolist()
    for i in range(0, len(src)):
        for j in range(0, len(src[i])):
            src[i][j] = 0.21 * src[i][j][0] + 0.72 * src[i][j][1] + 0.07 * src[i][j][2]
    return np.array(src)

#fold a sigmoid across an array
def sigmoid(array):
    ret = 1/(1 + np.exp(-array))
    return ret

#initiate images
def init_processing():
    # Read the image
    img = cv2.imread('imgs/original.jpg')
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    left = img[:, :width_cutoff]
    right = img[:, width_cutoff:]

    cv2.imwrite("./imgs/left.jpg", left)
    cv2.imwrite("./imgs/right.jpg", right)


    #gray scale
    grayLeft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)


    cv2.imwrite("./imgs/grayLeft.jpg", grayLeft)
    cv2.imwrite("./imgs/grayRight.jpg", grayRight)


def main():
    init_processing()

if __name__ == "__main__":
    main()

def init_processing():
    # Read the image
    img = cv2.imread('imgs/original.jpg')
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    left = img[:, :width_cutoff]
    right = img[:, width_cutoff:]

    cv2.imwrite("./imgs/left.jpg", left)
    cv2.imwrite("./imgs/right.jpg", right)


    #gray scale
    grayLeft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("./imgs/grayLeft.jpg", grayLeft)
    cv2.imwrite("./imgs/grayRight.jpg", grayRight)


def main():
    init_processing()

if __name__ == "__main__":
    main()