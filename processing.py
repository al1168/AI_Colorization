import cv2

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