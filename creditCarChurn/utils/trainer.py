from lightgbm import LGBMClassifier


class Estimator:
    """
        Sklearn-like estimator class for custome use
    """
    @staticmethod
    def fit(train_x, train_y):
        """
            Fit method for estimator to train model on given data
        Args:
            train_x (pandas.DataFrame): Features DataFrame
            train_y (pandas.DataFrame): Target DataFrame

        Returns:
            sklearn.estimator: trained estimator
        """
        return LGBMClassifier().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        """
            Compute target for given dataset from given trained estimator
        Args:
            trained (sklearn.estimator): Trained estimator 
            test_x (pandas.DataFrame): Features DataFrame

        Returns:
            numpy.ndarray: predicted value of given data from given estimator
        """
        return trained.predict(test_x)
