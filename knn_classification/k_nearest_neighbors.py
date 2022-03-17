# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Roopank Kohli] -- [rookohli]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

#### REFERENCES ####

# The below mentioned resources were referred to implement the solution for this specific problem.

# https://www.youtube.com/watch?v=wTF6vzS9fy4 - This video was referred to understand the intuition and explanation of
# the algorithm.

# https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/

# https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/ - referred to gather the formula for
# calculating euclidean distance

# https://www.statology.org/manhattan-distance-python/ - referred to gather the formula for calculating the manhattan
# distance


import numpy as np

from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        self._X = X
        self._y = y

        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        # raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        prediction = [self.makePrediction(x) for x in X]

        return np.array(prediction)

        # raise NotImplementedError('This function must be implemented by the student.')

    def makePrediction(self, sample):

        alpha = 0.000001

        # Calculating the distances of a sample test point from a point in training dataset and storing the values in
        # a list.

        distanceList = [euclidean_distance(sample, x_train) for x_train in self._X]

        # Extracting the sorted distances from the List. The number of distances extracted is equal to n_neighbors
        distances = np.sort(distanceList)[:self.n_neighbors]

        # Weights were calculated and and stored in a list. These weights are nothing but the inverse of distances.
        weightedList = [1 / (i + alpha) for i in distances]

        # Mapped indices are fetched using the distance List.
        index = np.argsort(distanceList)[:self.n_neighbors]

        # Corresponding labels for those indices were extracted from the training dataset.
        labels = [self._y[i] for i in index]

        # A dictionary is created to compute the weight for each label.
        weightDict = {}
        weight = 0
        for i in labels:

            if i not in weightDict.keys():
                weightDict[i] = weightedList[weight]
            else:
                weightDict[i] += weightedList[weight]
            weight += 1

        if self.weights == "uniform":
            item = max(labels, key=labels.count)
        else:
            item = max(weightDict, key=weightDict.get)

        return item
