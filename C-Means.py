import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = nodes, v = centroids, u = memberships, c = number of clusters, m = a parameter > 1

def norm(node1, node2):
    distance = 0
    for i in range(len(node1)):
        distance += (node1[i]-node2[i])**2
    return distance**0.5

#Step 1: calculate membership of each node
def membership_cal(x, u, v, m):
    for i in range(len(v)):
        for k in range(len(x)):
            if norm(x[k], v[i]) != 0:
                sigma = 0
                for j in range(len(v)):
                    sigma += (norm(x[k], v[i])/norm(x[k], v[j]))**(2/(m-1))
                u[i, k] = 1 / sigma
            else:
                u[i, k] = 1
                for j in range(len(v)):
                    if j != i:
                        u[j, k] = 0


#Step 2: calculate new centroids
def centroid_cal(x, v, u, m):
    for i in range(len(v)):
        temp1 = 0
        temp2 = 0
        for k in range(len(x)):
            temp1 += ((u[i, k])**m)*x[k]
            temp2 += (u[i, k])**m
        v[i] = temp1/temp2

#Calculate the cost
def cost(x, v, u, m):
    cost = 0
    for j in range(len(x)):
        for i in range(len(v)):
            cost += ((u[i][j])**m)*(norm(x[j], v[i])**2)
    return cost

def c_means(x, c, m):
    costs = 0
    # center of the clusters and memberships
    v = []
    # choose first centroids randomly and calculate memberships
    col_min = np.amin(x, axis=0)
    col_max = np.amax(x, axis=0)
    for i in range(c):
        node = []
        for j in range(len(col_min)):
            node.append(np.random.uniform(col_min[j], col_max[j]))
        v.append(node)
    # calculate membership and new centroids for 100 times
    u = np.zeros((len(v), len(x)))
    for i in range(100):
        membership_cal(x, u, v, m)
        centroid_cal(x, v, u, m)
    costs = cost(x, v, u, m)
    cluster_plot(x,u,v)
    return costs

#scatter points based on their membership size
def cluster_plot(x, u, v):
    xaxis = x[:,0]
    yaxis = x[:,1]
    v=np.array(v)
    for i in range(len(v)):
        plt.scatter(xaxis, yaxis,alpha=u[i])
    plt.scatter(v[:,0],v[:,1],color="purple",s=200,marker='^')
    plt.show()

#read from CSV data file
data = pd.read_csv("data1.csv", header = None)
data_arr = np.array(data.values)

costs = []
for i in range(1, 7):
    costs.append(c_means(data_arr, i, 2))

plt.plot(list(range(1,7)), costs)
plt.show()









