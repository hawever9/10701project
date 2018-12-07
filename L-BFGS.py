import numpy as np
import math 
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize

# Questions:
# L-BFGS implementation details, already changed to -obj, -pd
# eq(7) : what happens if one of theta_k's entries is zero in d Omega / d theta_k, use int.max?

# V : list of user id's, list of integers
# Vs: [[,], [,], ...], contains all possible edges, n*2 2-D Array
# E : all edges, list of tuples, undirected edges 
# C_t : a row is a binary indicator vector, one circle per row, len(each row) = number of users
# alpha_t : dim = k*1, alpha_t[k] = alpha_t_k = trade-off parameter for circle k
# theta_t : dim = k*F, theta_t[k] = theta_t_k = define circle property 
# theta_t_k : 1*F
# Fs: (n*n) * F, each row contains the feature vector for an edge, n*n edges in total 
# K : number of circles

# get V*V
def getAllPossE(V):
    res = []
    for v1 in V:
        for v2 in V:
            res.append([v1, v2])
    return np.asarray(res)
        
def d_k_e(u, v, C_t_k, alpha_t_k):
    return 1 if (C_t_k[u] == 1 and C_t_k[v] == 1) else -alpha_k

def getPhi(e, Fs, C_t, theta_t, alpha_t):
    Phi = 0
    n = len(C_t[0])
    (u, v) = (e[0], e[1])
    for k in range(len(C_t)):
        C_t_k = C_t[k]  # list of binary indicators
        alpha_t_k = alpha_t[k]
        theta_t_k = theta_t[k]
        d_k_e = d_k_e(u, v, C_t_k, alpha_t_k)
        Phi += d_k_e * (np.inner(Fs(u*n + v), theta_t_k))
    return Phi

def getLL(Vs, E, Fs, C_t, theta_t, alpha_t):
    term1 = 0
    term2 = 0
    n = len(C_t[0])
    for e in E:
        term1 += getPhi(e, Fs, C_t, theta_t, alpha_t)
    for e in Vs:
        term2 += math.log(1 + math.exp(getPhi(e, Fs(u*n + v), C_t, theta_t, alpha_t)))
    return term1 - term2

def getOmega(theta_t):
    return theta_t.sum()

def getObj(Vs, E, Fs, lbd, K, C_t, theta_t, alpha_t):
    L = getLL(Vs, E, Fs, C_t, theta_t, alpha_t)
    pen = getOmega(theta_t)
    return L - lbd * pen

def getOmegaPD(theta_t_k): # dim = 1*F
    pd = np.copy[theta_t_k]
    pd[pd < 0] = -1
    pd[pd > 0] = 1
    pd[pd == 0] = 0 
    return pd

def getPDtheta_k(Vs, E, Fs, C_t, theta_t, alpha_t, k): # ? vector
    term1 = 0
    term2 = 0
    n = len(C_t[0])
    (C_t_k, theta_t_k, alpha_t_k) = (C_t[k], theta_t[k], alpha_t[k])    
    for e in Vs:
        (u, v) = (e[0], e[1])
        Phi = getPhi(e, Fs, C_t, theta_t, alpha_t)
        term1 += (-d_k_e(u, v, C_t_k, alpha_t_k)) * Fs(u*n + v) * (math.exp(Phi)/(1 + math.exp(Phi))) # curr. dim = 1*F
    for e in E:
        (u, v) = (e[0], e[1])
        term2 += d_k_e(u, v, C_t_k, alpha_t_k) * Fs(u*n + v) # curr. dim = 1*F
    return term1 + term2 - getOmegaPD(theta_t_k) # dim = 1*F

def getPDalpha_k(Vs, E, Fs, C_t, theta_t, alpha_t, k): 
    term1 = 0
    term2 = 0
    n = len(C_t[0])
    (C_t_k, theta_t_k, alpha_t_k) = (C_t[k], theta_t[k], alpha_t[k])   
    for e in Vs:
        (u, v) = (e[0], e[1])
        if (C_t_k[u] == 1 and C_t_k[v] == 1):
            continue
        else:
            Phi = getPhi(e, Fs, C_t, theta_t, alpha_t)
            term1 += np.inner(Fs(u*n + v), theta_t_k) * (math.exp(Phi)/(1 + math.exp(Phi)))
    for e in E:
        (u, v) = (e[0], e[1])
        if (C_t_k[u] == 1 and C_t_k[v] == 1):
            continue
        else:
            term2 += np.inner(Fs(u*n + v), theta_t_k) 		
    return term1 - term2

def LBFGS(V, E, Fs, lbd, K, C_t, theta_t, alpha_t):
    Vs = getAllPossE(V)
    new_alpha = np.zeros(K)
    new_theta = np.zeros(theta_t.shape) # ? 
    obj = -1 * getObj(Vs, E, Fs, lbd, K, C_t, theta_t, alpha_t)
    for k in range(len(theta_t)):
        theta_t_k = theta_t[k]
        d_theta_k = -1 * getPDtheta_k(Vs, E, Fs, C_t, theta_t, alpha_t, k)
        # ? negative of objective function
        theta_new_k = optimize.fmin_l_bfgs_b(obj, d_theta_k) 

        # ? a new loop or optimize two params simultaneously
        d_alpha_k = -1 * getPDalpha_k(Vs, E, Fs, C_t, theta_t, alpha_t, k)
        alpha_new_k = optimize.fmin_l_bfgs_b(obj, d_theta_k)

        new_alpha[k] = alpha_new_k
        new_theta[k, : ] = theta_new_k
    return (new_alpha, new_theta)

def processCIRCLE(f):
    return np.loadtxt(f)

def processALPHA(f):
    return np.loadtxt(f)

def processTHETA(f):
    return np.loadtxt(f)

def processEdge(f):
    adj = np.loadtxt(f)
    return np.transpose(np.nonzero(adj))

def processVX(f):
    return np.loadtxt(f)

def processFeatures(f):
    return np.loadtxt(f)

def writeTHETA(f, new_theta):
    np.savetext(f, new_theta)
    return 

def writeALPHA(f, new_alpha):
    np.savetext(f, new_alpha)
    return 

if __name__ == '__main__':
    ego_id = "0"
    phi_type = "1"    

    theta_t = processTHETA("THETA.txt")
    alpha_t = processALPHA("ALPHA.txt")
    C_t = processCIRCLE("CIRCLE.txt")
    E = processEdge(ego_id + "_adja")
    V = processV(ego_id + "_vx")
    Fs = processFeatures(ego_id + "_phi" + phi_type)

#   (new_alpha, new_theta) = LBFGS(V, E, Fs, lbd, K, C_t, theta_t, alpha_t)

    writeALPHA("THETA.txt", new_alpha)
    writeTHETA("ALPHA.txt", new_theta)





    


