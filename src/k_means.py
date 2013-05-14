#coding:utf-8
import numpy as np
#----------------------------------------------------------------------


def kmeans(data, n_clusters, iterator=100, init_method="kmeans++", try_times=10):
    """
    kmeans algorithm

    parameters
    -----------------------------
    data: input data shape(n_samples,n_features)
    n_clusters: cluster number
    iterator: iteration number
    init_method: method to init centers,like random,kmeans++,default "kmeans++"

    return
    ------------------------------
    centers: last center ids, shape(k,n_features)
    inertia: sum distance to nearest center
    labels: input data's labels shape(n_samples,)
    """
    best_labels, best_inertia, best_centers = None, None, None
    for time in range(try_times):
        centers, inertia, labels = single_kmeans(data, n_clusters, iterator, init_method)
        print "candidate center:%s" % centers
        print "candidate inertia:%s" % inertia
        print "candidate labels:%s" % labels
        print "=================================="

        if best_inertia is None or inertia < best_inertia:
            best_centers = centers
            best_inertia = inertia
            best_labels = labels
    return best_centers, best_inertia, best_labels


def single_kmeans(data, n_clusters, iterator=100, init_method="kmeans++"):
    """
    single kmeans proccess

    return:
    --------------------------------------------------
    centers: last center ids, shape(k,n_features)
    inertia: sum distance to nearest center
    labels: input data's labels shape(n_samples,)

    """
    best_labels, best_inertia, best_centers = None, None, None
    n_samples = data.shape[0]

    centers = init_centers(data, n_clusters, init_method)

    for iter_index in range(iterator):
        centers_old = centers.copy()
        #keep each data pointer's distance to nearest center
        distances = np.zeros(shape=(data.shape[0],), dtype=np.float64)

        #E-step rearrange data to centers
        labels, inertia = e_kmeans(data, centers_old, n_clusters, distances)

        #M-step calculate new centers
        centers = m_kmeans(data, labels, n_clusters, distances)

        if best_inertia is None or inertia < best_inertia:
            best_centers = centers
            best_inertia = inertia
            best_labels = labels
    return best_centers, best_inertia, best_labels


def init_centers(data, n_clusters, init_method="kmeans++"):
    """
    center init

    parameters:
    -----------------------------------------------------
    data: input data set
    n_clusters: cluster number
    init_method: "random" or "kmeans++"

    return:
    -----------------------------------------------------
    centers: init centers shape(n_cluster,n_samples)
    """
    n_samples = data.shape[0]
    #random init center pointers
    if init_method == "random":
        seeds = np.random.permutation(n_samples)[:n_clusters]
        centers = data[seeds]
    else:
        centers = k_init(data, n_clusters)
    return centers;


def m_kmeans(data, labels, n_clusters, distances):
    """
    M_step in kmeans,will calculate new centers
    """
    n_sample = data.shape[0]
    n_features = data.shape[1]

    centers = np.zeros((n_clusters, n_features))
    #calculate  each cluster's data sample number
    #print labels
    sample_number_in_each_cluster = bincount(labels, minlength=n_clusters)
    #find empty cluster which don't have follow data samples
    empty_clusters = np.where(sample_number_in_each_cluster == 0)[0]

    if len(empty_clusters):
        print distances
        far_from_centers = distances.argsort()[::-1]
        print far_from_centers

    #rerrange a new data sample to empty cluster
    for i, cluster_id in enumerate(empty_clusters):
        new_center = data[far_from_centers[i]]
        centers[cluster_id] = new_center
        sample_number_in_each_cluster[cluster_id] = 1

    for sample_index in range(n_sample):
        for feature_index in range(n_features):
            centers[labels[sample_index], feature_index] += data[sample_index, feature_index]

    centers /= sample_number_in_each_cluster[:, np.newaxis]

    return centers


def e_kmeans(data, centers, n_clusters, distances):
    """
    E step of kmeans

    parameters:
    ------------------------------------------------
    data: input data set
    centers: cluster centers
    n_clusters: cluster number
    distances:each data pointer's distance to nearest center

    return:
    ------------------------------------------------
    inertia: sum distance to nearest center
    labels: input data's labels shape(n_samples,)
    """
    n_samples = data.shape[0]
    labels = -np.ones(n_samples, dtype=int)
    #sum of distances
    inertia = 0.0

    for sample_index in range(n_samples):
        min_distance = -1
        for cluster_index in range(n_clusters):
            distance = cal_distance(data[sample_index], centers[cluster_index])

            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                labels[sample_index] = cluster_index
                #store each sample's distance to nearest center
            distances[sample_index] = min_distance
            inertia += min_distance
    return labels, inertia


def k_init(data, n_clusters):
    """
    init centers use kmeans++
    """
    n_samples, n_features = data.shape

    centers = np.empty((n_clusters, n_features))
    center_id = np.random.randint(n_samples)

    centers[0] = data[center_id]

    for i in xrange(1, n_clusters):
        #calculate distances,nearest distance to the closest center
        distances = np.array([min([cal_distance(c, x) for c in centers]) for x in data])

        distances_probs = distances / distances.sum()
        cumsum_probs = distances_probs.cumsum()

        r = np.random.random()
        for index, prob in enumerate(cumsum_probs):
            if r < prob:
                target_index = index
                break
        centers[i] = data[target_index]
    return centers


def cal_distance(X, Y):
    """
    calculate two vector distance
    """
    distance = 0.0
    distance += ddot(X, X)
    distance += ddot(Y, Y)
    distance -= 2 * ddot(X, Y)

    return distance

#----------------------------------------------------------------------
def ddot(X, Y):
    """
    calculate vector dot of X and Y

    parameters
    ---------------------------------------
    X:vector,array like
    Y:vector,array like

    return
    ---------------------------------------
    dot of vector X and Y
    """
    return np.dot(X, Y)


def bincount(X, minlength=None):
    """
    calcalute each number's  times
    """
    result = np.bincount(X)
    if len(result) >= minlength:
        return result
    out = np.zeros(minlength, np.int)
    out[:len(result)] = result
    return out


def test():
    data = np.array([[1, 2], [3, 4], [4, 5], [10, 20], [11, 13], [12, 14], [20, 21], [22, 22]])
    centers, inertia, labels = kmeans(data, 3)
    print "target center:%s" % centers
    print "target inertia:%s" % inertia
    print "target:%s" % labels


if __name__ == "__main__":
    test()