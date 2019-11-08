from collections import Counter

import numpy as np


class KNeighborsClassifier:
    def __init__(self, k, X_train, y_train, weights=False):
        if not (isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray)):
            raise Exception('Wrong datatype, please pass numpy arrays')
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.columns_of_train = X_train.shape[1]
        self.weights = weights

    def fit(self):
        print('Training completed')

    def predict(self, testing_data):
        # Checking if the input is a numpy array with correct dimensions
        if len(testing_data.shape) > 1:
            columns_of_test = testing_data.shape[1]
        else:
            raise Exception('Please pass in a 2d array with same number of dimensions as in x_train')
        if columns_of_test != self.columns_of_train:
            raise Exception(
                'X_train column: {}\nTesting data column: {}'.format(self.X_train.shape[1], testing_data.shape[1]))
        if self.weights:
            return self.weightedPredict(testing_data)
        else:
            # Computing the distance of each testing data row with all other training data rows
            distance = dict()  # stores x_train_index:distance_from_testing_data
            index_position = 0  # Index of X_train (this is used to get the classification of the testing_data from y_train)
            classification = np.array([], dtype='int')

            for i in testing_data:
                for j in self.X_train:
                    distance[index_position] = np.sqrt(np.sum((i - j) ** 2))
                    index_position = index_position + 1
                distance = sorted(distance.items(), key=lambda x: (x[1], x[0]))
                classification = np.append(classification, [[self.classify(i, distance[0:self.k])]])
                distance = dict()
                index_position = 0
            return classification

    # to classify the testing_data, we pass only top k tuples with lowest distance from testing data

    def classify(self, test_data, distance):
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        label = Counter(label).most_common()
        return label[0][0]

    def weightedPredict(self, testing_data):
        distance = dict()  # stores x_train_index:distance_from_testing_data
        index_position = 0  # Index of X_train (this is used to get the classification of the testing_data from y_train)
        classification = np.array([], dtype='int')

        for i in testing_data:
            for j in self.X_train:
                distance[index_position] = np.sqrt(np.sum((i - j) ** 2))
                index_position = index_position + 1
            distance = sorted(distance.items(), key=lambda x: (x[1], x[0]))
            classification = np.append(classification, [[self.weightedClassify(i, distance[0:self.k])]])
            distance = dict()
            index_position = 0
        return classification

    def weightedClassify(self, test_data, distance):
        label = dict()
        for i in distance:
            label[self.y_train[i[0]]] = label.get(self.y_train[i[0]], 0) + (1 / i[1])
        label = sorted(label.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return label[0][0]
