import numpy as np
import matplotlib.pyplot as mpt
import random
def distance(x, y):
    return np.sqrt(np.sum(np.power(y-x ,2)))

def find_nearest_neighbors(p, points, k = 5):
    """
    find the k nearest neighbours of point p and return their indices.
    """
    distances = np.zeros(points.shape[0])
    for x in range(len(distances)):
        distances[x] = distance(p, points[x])
        
    ind = np.argsort(distances)
    return ind[:k]


def majority_votes(votes):
    """
    count votes and return winner
    """
    count_vote = {}
    
    for x in votes:
        count_vote[x] = count_vote.get(x, 0) + 1
    
 
    winners = []
    max_count = max(count_vote.values())
    
    for vote,count in count_vote.items():
        if count == max_count:
            winners.append(vote)
        
    return random.choice(winners)


def knn_predict(p, points, outcomes, k):
    ind = find_nearest_neighbors(p, points, k)
    
    return majority_votes(outcomes[ind])


points = np.array([[1, 1],[1, 2],[1, 3],[2, 1],[2, 2],[2, 3],[3, 1],[3, 2],[3, 3]])
p = np.array([2.5, 2])
outcomes = np.array([0,0,0,1,1,1,1,1])

print(knn_predict(p, outcomes, points, k = 3))
mpt.plot(points[:,0], points[:,1], "ro")
mpt.plot(p[0], p[1], "bo")
mpt.axis([0.5, 4, 0.5, 3.5])
