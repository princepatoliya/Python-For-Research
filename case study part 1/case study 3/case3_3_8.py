import numpy as np
import matplotlib.pyplot as mpt
import random
import scipy.stats as ss

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

def generate_synth_data(n = 50):
    """
    create two set of points from bivariate normal distributions.
    """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))

    return (points, outcomes)



def make_prediction_grid(predictors, outcomes, limits, h, k):
    """
    classify the each point on the prediction grid.
    """
    (x_min, x_max, y_min, y_max) = limits

    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)

    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype = int)

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x, y])

            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)

    return (xx, yy, prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    mpt.figure(figsize =(10,10))
    mpt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    mpt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    mpt.xlabel('Variable 1'); mpt.ylabel('Variable 2')
    mpt.xticks(()); mpt.yticks(())
    mpt.xlim (np.min(xx), np.max(xx))
    mpt.ylim (np.min(yy), np.max(yy))
    mpt.savefig(filename)

(predictors, outcomes) = generate_synth_data()

k = 5; filename = "knn_systh_5.pdf"; limits = (-3, 4, -3, 4); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

k = 50; filename = "knn_systh_50.pdf"; limits = (-3, 4, -3, 4); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


from sklearn import datasets

iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

mpt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")    
mpt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
mpt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")   
mpt.savefig("iris.pdf")

k = 5; filename = "iris_grid.pdf"; limits = (4, 8, 1.5, 4.5); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions =  knn.predict(predictors) 

