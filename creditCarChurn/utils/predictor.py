import pickle

from settings.constants import SAVED_ESTIMATOR


class Predictor:
    """
        Custom class to perfome prediction on saved estimator
    """

    def __init__(self):
        """
            Create an object of class with already saved and trained estimator
        """
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        """
            Compute target for given dataset
        Args:
            data (pandas.DataFrame): Features DataFrame

        Returns:
            numpy.ndarray: predicted value of given data
        """
        return self.loaded_estimator.predict(data)

    def predict_proba(self, data):
        """
            Compute probability of the target for given dataset
        Args:
            data (pandas.DataFrame): Features DataFrame

        Returns:
            numpy.ndarray: probability for predicted value of given data
        """
        return self.loaded_estimator.predict_proba(data)
