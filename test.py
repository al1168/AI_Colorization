# patch : 3 x 3 patch where the center pixel is the pixel we want to predict
# representative_colors: 5 colors that represents all the color in the original image
# grey_image: original image in a greyscaled
import time
from collections import Counter
from queue import PriorityQueue
import heapq
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import statistics
from statistics import mode
from scipy import stats
import random

def eclud_dist(centroid, data_pnt):
    return np.linalg.norm(centroid - data_pnt)


def predictpixel(representative_colors, grey_image):
    pass


def kmeans(input):
    colors = []
    id = list(range(0, len(input)))
    datapnts = {k: v for (k, v) in zip(id, input)}

    cluster_dict = {}
    for x in range(1, 6):
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        current = np.array([r, g, b])
        previous = np.array([-1, -1, -1])
        cluster_dict[x] = []
        cluster_dict[x].append(current)
        cluster_dict[x].append(previous)
    # while the current and previous dont equal each other, just keep averaging the data pnts related to cluster
    '''for each data pnt, assign it to a centroid, then when they are assigned, calculate average for each cluster, 
    and set as new centroid, check for convergence repeat'''
    final_clust_data = {}
    while not convergence(cluster_dict):
        # stores which datapnt belong to which cluster
        clust_data = {}

        for x in range(1, 6):
            clust_data[x] = []
        for datapnts_id in datapnts.keys():
            data = datapnts[datapnts_id]
            min_num = sys.maxsize
            min_cluster = -1
            # determine which cluster
            for key in cluster_dict.keys():
                value = cluster_dict[key]
                # print('COLOR '+str(data))
                temp = eclud_dist(value[0], data)
                if temp <= min_num:
                    min_cluster = key
                    min_num = temp
            clust_data[min_cluster].append(datapnts_id)
        # This is for replacing the current cluster pnt with the average of data pnts + replacing previous cluster pnt
        for key in clust_data.keys():
            value = clust_data[key]
            tmplst = []
            for data_id in value:
                tmplst.append(datapnts[data_id])
            a = tmplst
            if len(tmplst) > 0:
                average = np.mean(a, axis=0)
                # print(' Average of:' + str(a) + 'is ' + str(average))
                previous = cluster_dict[key][0]
                cluster_dict[key][0] = average
                cluster_dict[key][1] = previous
            else:
                previous = cluster_dict[key][0]
                cluster_dict[key][0] = np.array([-1, -1, -1])
                cluster_dict[key][1] = previous
        final_cluster_data = clust_data
    for key in cluster_dict.keys():
        colors.append(cluster_dict[key][0])
    print("finish" + str(len(colors)))
    return colors, final_cluster_data

    # while not np.array_equal(previous,current):


def convergence(cluster_dict):
    for value in cluster_dict.values():
        if not np.array_equal(value[0], value[1]):
            return False
    return True


# def main():
#     img = cv2.imread('imgs/left.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img[0:150, 0:300] = [0, 255, 0]
#
#     cluster_input = img.reshape((img.shape[1] * img.shape[0], 3))
#     colors, cluster_dict = kmeans(cluster_input)
#
#     h = img.shape[0]
#     w = img.shape[1]
#     for y in range(0, h):
#         for x in range(0, w):
#             min = sys.maxsize
#             tmpcolor = [0, 0, 0]
#             for color in colors:
#                 dist = eclud_dist(img[y, x], color)
#                 if min > dist:
#                     min = dist
#                     tmpcolor = color
#             img[y, x] = tmpcolor
#
#     plt.imshow(img)
#     plt.show()
#     # ///////////////////////////////////////////////////////////////////////
#     # recolor the right side of the image
#     '''
#     We analyze the greyscale of both the right and left, for each pixel on right side, take a patch surrounding
#     that pixel and find 6 patches on the left side most similar to that pixel
#     '''
#     greyR = cv2.imread('imgs/grayRight.jpg')
#     greyL = cv2.imread('imgs/grayLeft.jpg')
#     greydict = {}
#     h = greyL.shape[0]
#     w = greyL.shape[1]
#     # assign neighbors for all the leftside
#     for i in range(2, h - 1):
#         for j in range(2, w - 1, 3):
#             tmplst = [greyL[i + 1, j],
#                       greyL[i - 1, j],
#                       greyL[i, j + 1],
#                       greyL[i, j - 1],
#                       greyL[i - 1, j - 1],
#                       greyL[i + 1, j - 1],
#                       greyL[i - 1, j + 1],
#                       greyL[i + 1, j + 1],
#                       greyL[i, j]]
#             flat_list = np.array(np.concatenate(tmplst).flat)
#             greydict[(i, j)] = flat_list
#     o = greyR.shape[0]
#     p = greyR.shape[1]
#     # compute each pixel and predict color
#     for i in range(2, o - 1):
#         for j in range(2, p - 1):
#             print(i)
#             tmplst = [greyR[i + 1, j],
#                       greyR[i - 1, j],
#                       greyR[i, j + 1],
#                       greyR[i, j - 1],
#                       greyR[i - 1, j - 1],
#                       greyR[i + 1, j - 1],
#                       greyR[i - 1, j + 1],
#                       greyR[i + 1, j + 1],
#                       greyR[i, j]]
#             flat_list = np.array(np.concatenate(tmplst).flat)
#             pqe = PriorityQueue()
#             for key in greydict.keys():
#                 dist = eclud_dist(flat_list, greydict[key])
#                 pqe.put((dist, key))
#             clst = []
#             nclst = []
#             priolst = []
#             for x in range(6):
#                 patch = pqe.get()
#                 q, z = patch[1]
#                 color = img[q, z]
#                 clst.append(color)
#                 print(color)
#                 print(clst)
#             m = stats.mode(clst)
#             mode1 = m[0]
#             for x in range(len(clst)):
#                 comparison = clst[x] == m[0]
#                 equal_arrays = comparison.all()
#                 # print(equal_arrays)
#                 if not equal_arrays:
#                     clst.append(clst[x])
#             y = stats.mode(nclst)
#             mode2 = y[0]
#             if y[1] == m[1]:
#                 for x in range(len(clst)):
#                     popped = clst.pop()
#                     if popped == mode1:
#                         greyR[i, j] = mode2
#                     if popped == mode2:
#                         greyR[i, j] = mode1
#                         break
#             else:
#                 greyR[i, j] = mode1
#
#     plt.imshow(greyR)
#     plt.show()
#     lst = []
#     # lst = [np.array([0,0,0]),np.array([255,255,255]),np.array([51,51,51]),np.array([51,51,51]),np.array([51,51,51]),np.array([51,51,51]),np.array([102,102,102]),np.array([153,153,153]),np.array([0,0,153]),np.array([0,0,153]),np.array([0,0,153])]
#     # m = stats.mode(lst)
#     # flat_list = np.array(np.concatenate(m[0]).flat)
#     # # print(flat_list)
#     # nlst = []
#     # # for arr in lst:
#     # #     print(arr)
#     # #     comparison = arr == m[0]
#     # #     # print(comparison)
#     # #     equal_arrays = comparison.all()
#     # #     print(equal_arrays)
#     # #     # if not equal_arrays:
#     # #     #     lst.append(arr)
#     # for x in range(len(lst)):
#     #     comparison = lst[x] == m[0]
#     #     equal_arrays = comparison.all()
#     #     # print(equal_arrays)
#     #     if not equal_arrays:
#     #         nlst.append(lst[x])
#     # y = stats.mode(nlst)
#
#     # for x in range(10):
#     #     lst.append(np.random.randint(low=0, high=255, size=3))
#     # print(lst)
#     # print(kmeans(lst))

def get_patches(img):
    patches=[]
    #iterate through grayleft
    #iterate through rows
    for i in range(1,len(img)-1):
        #iterate through columns
        for j in range(1,len(img[0])-1):
            patches.append((img[i-1:i+2,j-1:j+2],(i,j)))

    return patches
def main():
    img = cv2.imread('imgs/left.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img[0:150, 0:300] = [0, 255, 0]

    cluster_input = img.reshape((img.shape[1] * img.shape[0], 3))
    img2 = cv2.imread('imgs/right.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    colors, cluster_dict = kmeans(cluster_input)

    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            min = sys.maxsize
            tmpcolor = [0, 0, 0]
            for color in colors:
                dist = eclud_dist(img[y, x], color)
                if min > dist:
                    min = dist
                    tmpcolor = color
            img[y, x] = tmpcolor

    h = img2.shape[0]
    w = img2.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            min = sys.maxsize
            tmpcolor = [0, 0, 0]
            for color in colors:
                dist = eclud_dist(img2[y, x], color)
                if min > dist:
                    min = dist
                    tmpcolor = color
            img2[y, x] = tmpcolor
    plt.imshow(combine(img,img2))
    plt.show()

    # ///////////////////////////////////////////////////////////////////////
    # recolor the right side of the image
    '''
    We analyze the greyscale of both the right and left, for each pixel on right side, take a patch surrounding
    that pixel and find 6 patches on the left side most similar to that pixel
    '''
    greyR = cv2.imread('imgs/grayRight.jpg')
    greyL = cv2.imread('imgs/grayLeft.jpg')
    greyLp = get_patches(greyL)

    for i in range(1, len(greyR) - 1):
        # iterate through columns
        for j in range(1, len(greyR[0]) - 1):
            print(i)
            patch = greyR[i - 1:i + 2, j - 1:j + 2]
            samples = random.sample(list(greyLp), 1000)
            pqe = PriorityQueue()
            for sample in samples:
                dist = eclud_dist(sample[0], greyR[i - 1:i + 2, j - 1:j + 2])
                pqe.put((dist,sample[1]))
            closestNeighbor = []
            for x in range(6):
                popped = pqe.get()
                closestNeighbor.append(popped)
            # print(closestNeighbor)
            cl = []
            for patch in closestNeighbor:
                x, y = patch[1]
                color = img[x,y]
                cl.append(color)
            mode1 = stats.mode(cl)
            mode1color = mode1[0]
            print("mode1 =" + str(mode1))
            nlst = []
            for patch in closestNeighbor:
                x, y = patch[1]
                color = img[x, y]
                comparison = color == mode1color
                equal_arrays = comparison.all()
                if not equal_arrays:
                    nlst.append(color)
            mode2 = stats.mode(nlst)
            mode2color = mode2[0]
            print("cnt of mode1"+ str(mode1[1])+"\tcnt of mode2:" +str(mode2[1]))
            if len(mode2[1]) > 0:
                cnt1 = np.array(np.concatenate(mode1[1]).flat)
                cnt2 = np.array(np.concatenate(mode2[1]).flat)

                comp = cnt1 == cnt2
                e = comp.all()
                print(e)
                if e:
                    for ptr in closestNeighbor:
                        popped = closestNeighbor.pop()
                        if popped == mode1color:
                            greyR[i,j] = mode2color
                            break
                        if popped == mode2color:
                            greyR[i,j] = mode1color
            else:
                greyR[i,j] = mode1color

    plt.imshow(combine(img,greyR))
    plt.show()
    calculatediff(img2,greyR)

def combine(left,right):
    combined = []
    for i in range(0, len(left)):
        combined.append(list(left[i]) + list(right[i]))
    return combined

def calculatediff(img,predicted_image):
    numPixel = len(img)*len(img[1])
    correct = 0
    for i in range(0,len(img)):
        for j in range(0,len(img[1])):
            comparison = img[i,j] == predicted_image[i,j]
            equal_arrays = comparison.all()
            if equal_arrays:
                correct +=1
    print(str(1-(correct/numPixel))+" correctness")

if __name__ == "__main__":
    main()
