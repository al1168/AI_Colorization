import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL.ImageOps import scale

from basic import *
from processing import *

#train and build the model
def create_model(left):

    #find derivative between hidden and output layers
    def cost_d_1(input, layer, hidden_layer, output_layer, weigh1, weight2):
        a = 2 * np.diagflat(output_layer - np.transpose(np.array([layer])))
        b = np.exp(-np.dot(weight2, hidden_layer)) / np.square(1 + np.exp(-np.dot(weight2, hidden_layer)))

        dot_prod = np.dot(a, b)
        hidden_layer = np.transpose(hidden_layer)
        d = np.dot(dot_prod, hidden_layer)

        return d

    #find derivative between hidden and output layers
    def cost_d_2(input, layer, hidden_layer, output_layer, weight1, weight2):
        val = np.exp(-np.dot(weight1, np.transpose(input))) / np.square(
            1 + np.exp(-np.dot(weight1, np.transpose(input))))
        dot_prod = np.dot(val, input)
        sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(3):
            scalar = 2 * (output_layer[i] - layer[i]) * np.exp(-np.dot(weight2[i], hidden_layer)) / np.square(
                1 + np.exp(-np.dot(weight2[i], hidden_layer)))
            sum += (scalar * weight2[i])

        sum = np.diagflat(sum)
        d = np.dot(sum, dot_prod)

        return d

    left_grey = scale(left)

    p = get_patches(left_grey)
    np.random.shuffle(p)
    weight1 = np.random.rand(5,9)
    weight2 = np.random.rand(3,5)
    groups = []
    group_num = 200

    for i in range(group_num):
        groups.append(p[int(i * len(p) / group_num):int(len(p) / group_num * (1 + i) - 1)])
    for group in groups:
        for i in range(100):
            tot1 = np.zeros((3,5), dtype = float)
            tot2 = np.zeros((5,9), dtype = float)
            for patch in group:
                input = patch[0]
                layer = left[patch[1]]
                input = np.array([input.flatten()]) / 255
                layer = layer / 255
                #print(weight1.shape)
                #print(np.transpose(input_data).shape)
                hidden_layer = sigmoid(np.dot(weight1, np.transpose(input)))
                output_layer = sigmoid(np.dot(weight2, hidden_layer))

                cost = 0
                for a in range(len(output_layer)):
                    cost += np.square(output_layer[a] - layer[a])

                weight_calc_1 = cost_d_1(input, layer, hidden_layer, output_layer, weight1, weight2)
                weight_calc_2 = cost_d_2(input, layer, hidden_layer, output_layer, weight1, weight2)

                tot1 += weight_calc_1
                tot2 += weight_calc_2

            mat1 = tot1 / len(group)
            mat2 = tot2 / len(group)
            weight1 -= mat2
            weight2 -= mat1

    return weight1, weight2


#apply model to color right
def apply_model(weight1,weight2,img):
    greyR = scale(img)
    patches = get_patches(greyR)

    for i in patches:
        hidden_layer = sigmoid(np.dot(weight1, np.transpose(i[0].flatten()/255)))
        output_layer = sigmoid(np.dot(weight2, hidden_layer))
        img[i[1]] = output_layer * 255

    return img

#main
def main():

    left = cv2.imread('imgs/left.jpg')
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    right = cv2.imread('imgs/right.jpg')
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    left_gray = cv2.imread('imgs/grayLeft.jpg')
    left_gray = cv2.cvtColor(left_gray, cv2.COLOR_BGR2RGB)

    right_gray = cv2.imread('imgs/grayRight.jpg')
    right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2RGB)

    weight1,weight2 = create_model(left)
    ret = apply_model(weight1, weight2, right_gray)

    plt.imshow(combine(left,ret))
    plt.show()

if __name__ == "__main__":
    main()
