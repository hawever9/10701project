import numpy as np

# The index for edge is Edge - EgoEdge
# i.e current EgoEdge is 1684, want to check edge 1789, the index for it is x = 1789 - 1684
# Phi and Psy are n x n x d np array, with Phi(x, y) = Phi[x][y]
def getFeatures(ego_id):
    feat = np.loadtxt(ego_id + ".feat")
    featnames = open(ego_id + ".featnames").readlines()
    egofeat = np.loadtxt(ego_id + ".egofeat")

    # eliminate the user ID at the first column
    feat = np.delete(feat, 0, axis=1)
    # add ego user to the first row
    feat = np.concatenate((egofeat[np.newaxis, : ], feat))
    # number of users
    n = len(feat)
    # number of features
    d = len(feat[0])

    # get difference feature vector 1 (version without summarize)
    sigma1 = np.empty((n, n, d))
    for i in range(n):
        for j in range(n):
            sigma1[i][j] = feat[i] + feat[j] > 1


    indices = getGroupIndices(featnames)
    # get difference feature vector 2 (version with summarize)
    sigma2 = np.empty((n, n, len(indices)))
    for i in range(n):
        for j in range(n):
            for p in range(len(indices)):
                if p < len(indices) - 1:
                    sigma2[i][j][p] = np.sum(sigma1[i][j][indices[p]:indices[p+1]])
                else:
                    sigma2[i][j][p] = np.sum(sigma1[i][j][indices[p]:])

   
    phi1 = np.concatenate((np.ones((n,n,1)), -sigma1), axis=2)
    psi1 = np.concatenate((np.ones((n,n,1)), -sigma2), axis=2)
    phi2 = np.empty((n, n, d+1))
    psi2 = np.empty((n, n, len(indices)+1))
    for i in range(n):
        for j in range(n):
            phi2[i][j] = np.concatenate((np.asarray([1]), -np.abs(sigma1[i][0] - sigma1[j][0])), axis=0)
            psi2[i][j] = np.concatenate((np.asarray([1]), -np.abs(sigma2[i][0] - sigma2[j][0])),axis=0)

    return (phi1, phi2, psi1, psi2)




def getGroupIndices(featnames):
    indices = []
    prev_cat = ""
    for line in featnames:
        piece = line.split(' ')
        categories = piece[1]
        # meet a new category,
        if categories != prev_cat:
            # record the index
            indices.append(int(piece[0]))
            # update prev_cat
            prev_cat = categories
    return indices


# dim = max_edge + 1
# The index for edge is Edge - EgoEdge
# i.e current EgoEdge is 1684, want to check edge 1789, the index for it is x = 1789 - 1684
def getEdges(ego_id):
    edges = np.loadtxt(ego_id+".edges")
    dim = int(np.max(edges)-int(ego_id))
    #print(edges.shape)
    adja =  np.zeros((dim+1, dim+1))
    for (i, j) in edges:
        adja[int(i)-int(ego_id)][int(j)-int(ego_id)] = 1
    return adja



ID = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
for i in ID:
    getFeatures(str(i))
    getEdges(str(i))


