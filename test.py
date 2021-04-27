# patch : 3 x 3 patch where the center pixel is the pixel we want to predict
# representative_colors: 5 colors that represents all the color in the original image
# grey_image: original image in a greyscaled
import numpy as np
import sys


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

    while not convergence(cluster_dict):
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
                print(' Average of:' + str(a) + 'is ' + str(average))
                previous = cluster_dict[key][0]
                cluster_dict[key][0] = average
                cluster_dict[key][1] = previous
            else:
                previous = cluster_dict[key][0]
                cluster_dict[key][0] = np.array([-1, -1, -1])
                cluster_dict[key][1] = previous
    for key in cluster_dict.keys():
        colors.append(cluster_dict[key][0])
    print("finish" + str(len(colors)))
    return colors

    # while not np.array_equal(previous,current):


def convergence(cluster_dict):
    for value in cluster_dict.values():
        if not np.array_equal(value[0], value[1]):
            return False
    return True


def main():
    lst = []
    # lst = [np.array([0,0,0]),np.array([255,255,255]),np.array([51,51,51]),np.array([102,102,102]),np.array([153,153,153])]
    for x in range(10):
        lst.append(np.random.randint(low=0, high=255, size=3))
    print(lst)
    print(kmeans(lst))


if __name__ == "__main__":
    main()
