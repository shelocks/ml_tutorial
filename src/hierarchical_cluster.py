__author__ = 'shelocks'

import numpy as np


class cluster_node(object):
    """
    cluster node represent a cluster in hierarchical cluster algorithm

    parameters:
    ------------------------------------------------------------------
    cluster_vector: a vector to represent this cluster
    left:left child cluster node of this cluster
    right:right child cluster node of this cluster
    cluster_id: cluster's id
    """

    def __init__(self, cluster_vector, left=None, right=None, cluster_id=None):
        self.cluster_vector = cluster_vector
        self.left = left
        self.right = right
        self.cluster_id = cluster_id

    def __str__(self):
        left_node = ""
        right_node = ""
        if self.left is not None:
            left_node = self.left.__str__()
        if self.right is not None:
            right_node = self.right.__str__()

        return "[cluster cluster_id:%d,(left:%s,right:%s)]" % (self.cluster_id, left_node, right_node)


def distance(X, Y):
    """
    calculate distance between vector X and Y,you can implement
    other distance metrics

    parameters:
    ----------------------------------------------------------
    X:vector X
    Y:vector Y

    return:
    ----------------------------------------------------------
    distance between vector X and Y
    """
    return np.sqrt(np.sum((X - Y) ** 2))


def cal_cluster_vector(cluster_x, cluster_y):
    """
    calculate new cluster's cluster vector

    parameters:
    ----------------------------------------------------------
    cluster_x:cluster x
    cluster_y:cluster y

    return:
    ----------------------------------------------------------
    return a vector to represent the new cluster
    """
    return (cluster_x.cluster_vector + cluster_y.cluster_vector) / 2.0


def hierarchical_cluster(data):
    """
    cluster data use hierarchical cluster algorithm

    parameters:
    ----------------------------------------------------------
    data: data need to be clustering
    """
    n_sample = data.shape[0]

    #init clusters each data sample is a cluster
    clusters = [cluster_node(np.array(data[x]), cluster_id=x) for x in range(n_sample)]

    #hierarchical cluster proccess
    current_cluster_id = -1
    while len(clusters) > 1:
        nearest_cluster = (0, 1)
        min_distance = distance(clusters[0].cluster_vector, \
                                clusters[1].cluster_vector)

        #find two most nearest cluster
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_distance = distance(clusters[i].cluster_vector, \
                                            clusters[j].cluster_vector)
                if cluster_distance < min_distance:
                    nearest_cluster = (i, j)
                    min_distance = cluster_distance

        #calculate new cluster's vector
        cluster_vector = cal_cluster_vector(clusters[nearest_cluster[0]], clusters[nearest_cluster[1]])
        #new cluster
        new_cluster = cluster_node(cluster_vector, clusters[nearest_cluster[0]], clusters[nearest_cluster[1]], \
                                   current_cluster_id)

        current_cluster_id -= 1
        #remove previous cluster
        del clusters[nearest_cluster[1]]
        del clusters[nearest_cluster[0]]

        #add new cluster
        clusters.append(new_cluster)
    return clusters[0]


def test():
    data = np.array([[1, 2], [3, 4], [4, 5], [10, 20], [11, 13], [12, 14], [20, 21], [22, 22]])
    print hierarchical_cluster(data)


if __name__ == "__main__":
    test()
