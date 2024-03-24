import numpy as numpy
from collections import Counter

def calculate_euclidean(x1_val, x2_val):
    distance = numpy.sqrt(numpy.sum((x1_val - x2_val)**2))
    return distance

class KNN_Classifier:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def fit_data(self, X_data, y_labels):
        self.X_training = X_data
        self.y_training = y_labels

    def predict_data(self, X_data):
        predictions = [self._predict_single(x) for x in X_data]
        return predictions

    def _predict_single(self, x_val):
        # compute the distance
        distances = [calculate_euclidean(x_val, x_train_val) for x_train_val in self.X_training]
    
        # get the closest k
        k_indices = numpy.argsort(distances)[:self.num_neighbors]
        k_nearest_labels = [self.y_training[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
