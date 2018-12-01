import numpy as np
import math 
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize

# Questions:
# how to incorporate F, need to get V and E 
# Is there a difference between little_phi(e) and little_phi(e)_k, eq(7) in paper
# what is d Omega / d theta_k, eq(7) in paper
# L-BFGS implementation details


# V : integer, number of users 
# Vs: [(,), (,), ...], contains all possible edges
# E : all edges, list of tuples, undirected edges 
# C_t : a row is a binary indicator vector, one circle per row, len(each row) = number of users
# alpha_t : dim = k*1, alpha_t[k] = alpha_t_k = trade-off parameter for circle k
# theta_t : dim = k*F, theta_t[k] = theta_t_k = define circle property 
# theta_t_k : F * 1
# F(e): 1 * F

# output edge features little_phi(e), 1-d np array, dim : 1*F
def F(e): 
	pass

# get V*V
def getAllPossE(V):
	pass

def d_k_e(u, v, C_t_k, alpha_t_k):
	return 1 if (C_t_k[u] == 1 and C_t_k[v] == 1) else -alpha_k

def getPhi(e, F, C_t, theta_t, alpha_t):
	Phi = 0
    (u, v) = (e[0], e[1])
    for k in range(len(C_t)):
    	C_t_k = C_t[k]  # list of binary indicators
    	alpha_t_k = alpha_t[k]
    	theta_t_k = np.transpose(theta_t[k])
    	d_k_e = d_k_e(u, v, C_t_k, alpha_t_k)
    	Phi += d_k_e * (np.dot(F(e), theta_t_k))
    return Phi

def getLL(Vs, E, F, C_t, theta_t, alpha_t):
	term1 = 0
	term2 = 0
	for e in E:
		term1 += getPhi(e, F, C_t, theta_t, alpha_t)
    for e in Vs:
        term2 += math.log(1 + math.exp(getPhi(e, F, C_t, theta_t, alpha_t)))
	return term1 - term2

def getOmega(theta_t):
	return theta_t.sum()

def getObj(Vs, E, F, lbd, K, C_t, theta_t, alpha_t):
    L = getLL(Vs, E, F, C_t, theta_t, alpha_t)
    pen = getOmega(theta_t)
    return L-lbd * pen

def getPDtheta_k(Vs, E, F, C_t, theta_t, alpha_t, k): # ? vector
    term1 = 0
    term2 = 0
	(C_t_k, theta_t_k, alpha_t_k) = (C_t[k], np.transpose(theta_t[k]), alpha_t[k])    
	for e in Vs:
		(u, v) = (e[0], e[1])
		Phi = getPhi(e, F, C_t, theta_t, alpha_t)
        term1 += (-d_k_e(u, v, C_t_k, alpha_t_k)) * F(e) * (math.exp(Phi)/(1 + math.exp(Phi))) # ? 
    for e in E:
    	(u, v) = (e[0], e[1])
    	term2 += d_k_e(u, v, C_t_k, alpha_t_k) * F(e)
    return term1 + term2 - np.asarray([1] * len(theta_t_k[0])) # ?

def getPDalpha_k(Vs, E, F, C_t, theta_t, alpha_t, k): 
	term1 = 0
	term2 = 0
	(C_t_k, theta_t_k, alpha_t_k) = (C_t[k], np.transpose(theta_t[k]), alpha_t[k])   
	for e in Vs:
		(u, v) = (e[0], e[1])
		if (C_t_k[u] == 1 and C_t_k[v] == 1):
			continue
		else:
			Phi = getPhi(e, F, C_t, theta_t, alpha_t)
			term1 += np.dot(F(e), theta_t_k) * (math.exp(Phi)/(1 + math.exp(Phi)))
	for e in E:
		(u, v) = (e[0], e[1])
		if (C_t_k[u] == 1 and C_t_k[v] == 1):
			continue
		else:
			term2 += np.dot(F(e), theta_t_k) 		
	return term1 - term2

def LBFGS(V, E, F, lbd, K, C_t, theta_t, alpha_t):
	Vs = getAllPossE(V)
	new_alpha = np.zeros(K)
    new_theta = np.zeros(theta_t.shape) # ? 
    obj = getObj(Vs, E, F, lbd, K, C_t, theta_t, alpha_t)
    for k in range(len(theta_t)):
    	theta_t_k = np.transpose(theta_t[k])
        d_theta_k = getPDtheta_k(Vs, E, F, C_t, theta_t, alpha_t, k)
        # ? negative of objective function
        theta_new_k = optimize.fmin_l_bfgs_b(obj, d_theta_k) 

        # ? a new loop or optimize two params simultaneously
        d_alpha_k = getPDalpha_k(Vs, E, F, C_t, theta_t, alpha_t, k)
        alpha_new_k = optimize.fmin_l_bfgs_b(obj, d_theta_k)

        new_alpha[k] = alpha_new_k
        new_theta[k, : ] = np.transpose(theta_new_k)

    return (new_alpha, new_theta)

if __name__ == '__main__':
	# need to preprocess E, V


    


