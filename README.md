# K-Nearest-Neighbors-Classification-From-Scratch

## Problem Description

k-nearest neighbors is a type of non-parametric supervised machine learning algorithm that is used for both classification and regression tasks. For classification, the principle behind k-nearest neighbors is to find k training samples that are closest in distance to a new sample in the test dataset, and then make a prediction based on those samples. The objective in this part is to implement a k-nearest neighbors classifier from scratch.

## Solution and Approach

In this given problem, we were provided with two sets of datasets namely Iris dataset and Digits dataset, having distinct set of features associated with them. When a prediction is required, the k-most similar records to a new record from the training dataset are then located. From these neighbors, a summarized prediction is made. Similarity between records can be measured many ways. Here, I have used two distinct methods of measuring or locating the neighbors - Euclidean distance and Manhattan distance. Once the neighbors are discovered, the summary prediction can be made by returning the most common outcome or taking the average. As such, KNN can be used for classification or regression problems.

The code is implemented in two parts. First part is to train the classifier with the help of training data and second one is to test the classifier by predicting the results using the testing data.

While training the classifier, two numpy arrays were passed to the fit method. The first array X represented the input dataset and the second array Y contained the true class values for each sample in the input data.
These training data were fed to the classifier and based on it, testing was done. 

After that, the process of prediction was initiated in which first the distance (euclidean or manhattan) of a data point from the other neighboring data points was computed and stored.
Then after sorting the list, k nearest neighbors were collected. The final step was to check the class or category of those neighbouring data points, the one with the maximum probability was selected as the most suitable category, and therefore we were able to conclude that our testing data point belonged to that category.

## Design Decisions

A number of data structures including list, array, dictionary etc. and a variety of inbuilt functions like 
np.sort(), max(), np.argsort(), along with list comprehensions were used in this solution. The main approach to decide the designing decisions for this problem was to implement 
the solution in minimum time complexity and provide maximum accuracy. The output here displays the accuracy attained by our classifier on this specific
dataset which comes out to be more than 90%. 

## Assumpltions

It was assumed that the input data features are all numerical features and that the target class values are categorical features.

## Problems Faced

* It was found that the running time for digits dataset was comparatively longer than the iris dataset.

* In order to avoid division by zero error, a small value(alpha) was added to the denominator while calculating the weights.

