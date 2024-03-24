import numpy as numpy

class NaiveBayes:

    def fit_model(self, X_data, y_labels):
        n_samples, n_features = X_data.shape
        self._categories = numpy.unique(y_labels)
        n_categories = len(self._categories)

        # calculate mean, var, and prior for each category
        self._mean = numpy.zeros((n_categories, n_features), dtype=numpy.float64)
        self._var = numpy.zeros((n_categories, n_features), dtype=numpy.float64)
        self._priors = numpy.zeros(n_categories, dtype=numpy.float64)

        for idx, cat in enumerate(self._categories):
            X_cat = X_data[y_labels == cat]
            self._mean[idx, :] = X_cat.mean(axis=0)
            self._var[idx, :] = X_cat.var(axis=0)
            self._priors[idx] = X_cat.shape[0] / float(n_samples)
            

    def predict_model(self, X_data):
        y_pred = [self._predict_sample(x) for x in X_data]
        return numpy.array(y_pred)

    def _predict_sample(self, x):
        posteriors = []

        # calculate posterior probability for each category
        for idx, cat in enumerate(self._categories):
            prior = numpy.log(self._priors[idx])
            posterior = numpy.sum(numpy.log(self._pdf_distribution(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return category with the highest posterior
        return self._categories[numpy.argmax(posteriors)]

    def _pdf_distribution(self, category_idx, x):
        mean = self._mean[category_idx]
        var = self._var[category_idx]
        numerator = numpy.exp(-((x - mean) ** 2) / (2 * var))
        denominator = numpy.sqrt(2 * numpy.pi * var)
        return numerator / denominator


# Driver code for testing
if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def calculate_accuracy(y_true, y_pred):
        accuracy = numpy.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit_model(X_train, y_train)
    predictions = nb.predict_model(X_test)

    print("Naive Bayes classification accuracy", calculate_accuracy(y_test, predictions))
