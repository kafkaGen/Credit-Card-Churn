import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer


class DataLoader(object):
    """
        Class that take and do all preprocessing step with data so it can be used further in machine learning models
    """

    def fit(self, dataset):
        """
            Initialize dataset, create variable to store Label Encoder estimators
        Args:
            dataset (pandas.DataFrame): raw data in pandas DataFrame type
        """

        self.dataset = dataset.copy()
        self.le = dict()

    def encode_cat_with_unknown(self, feature):
        """
            Encode features with object type, replace 'Unknown' values with np.nan, save encoder estimator
        Args:
            feature (string): name of the categorical feature in dataset
        """

        le = LabelEncoder()
        self.dataset[feature] = le.fit_transform(self.dataset[feature])

        miss = np.argwhere(le.classes_ == 'Unknown')[0][0]
        self.dataset[feature] = self.dataset[feature].astype(np.float64)
        self.dataset.loc[self.dataset[feature] == miss, feature] = np.nan

        self.le[feature] = le

    def decode_cat_with_unknown(self, feature):
        """
            Decode features to categorical via saved estimator
        Args:
            feature (string): name of the categorical feature in dataset
        """

        self.dataset[feature] = self.le[feature].inverse_transform(
            self.dataset[feature])

    def load_data(self):
        """_summary_
            This function designed to do all the preprocessing step with data, so it can be used for machine learning
        Returns:
            X, y ((pd.DataFrame, pd.DataFrame)): X - preprocessed features of dataset, y - preprocessed target of dataset
        """
        # binarize target feature
        self.dataset['Target'] = np.where(
            self.dataset['Attrition_Flag'] == 'Existing Customer', 1, 0)
        # binarize Gender feature
        self.dataset['IsFemale'] = np.where(
            self.dataset['Gender'] == 'F', 1, 0)

        # binning with cut Customer Age
        self.dataset['Customer_Age'], self.age_bins = pd.qcut(
            self.dataset['Customer_Age'], 10, labels=np.arange(10), retbins=True)

        # encode Card Category feature with order
        self.dataset['Card_Category'] = self.dataset['Card_Category'].replace(
            ['Blue', 'Silver', 'Gold', 'Platinum'], [0, 1, 2, 3])

        # drop unnecessary features
        self.dataset = self.dataset.drop(['CLIENTNUM', 'Attrition_Flag', 'Gender',
                                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)

        # encode categorical features
        need_impute = ['Education_Level', 'Marital_Status', 'Income_Category']
        for feature in need_impute:
            self.encode_cat_with_unknown(feature)
        # impute all (in our case only categorical) features
        self.knni = KNNImputer(missing_values=np.nan,
                               n_neighbors=7, weights='distance').fit(self.dataset)
        self.dataset = pd.DataFrame(self.knni.transform(
            self.dataset), columns=self.dataset.columns)
        # after imputing categorical features we get float values, change and round it to integer
        self.dataset[need_impute] = self.dataset[need_impute].astype(np.int32)

        # set int type to integer features
        int_type = ['Customer_Age', 'Dependent_count', 'Card_Category', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Target', 'IsFemale']
        self.dataset[int_type] = self.dataset[int_type].astype(np.int32)

        # scale float features
        float_type = ['Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                      'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        self.standard_scaler = StandardScaler()
        self.dataset[float_type] = self.standard_scaler.fit_transform(
            self.dataset[float_type])

        # split dataset on features and target
        X = self.dataset.drop('Target', axis=1)
        y = self.dataset['Target'].copy()

        return X, y

    def one_transform(self, sample):
        """
            This function designed to do all the preprocessing step with one example of data, so it can be used for prediciton
        Args:
            sample (_type_): _description_

        Returns:
            X (pd.DataFrame): X - preprocessed features of dataset
        """
        # create dummy target variable
        sample['Target'] = 1
        # binarize Gender feature
        sample['IsFemale'] = np.where(sample['Gender'] == 'F', 1, 0)
        # drop unnecessary features
        sample = sample.drop('Gender', axis=1)

        # binning with cut Customer Age
        sample['Customer_Age'] = np.searchsorted(
            self.age_bins, sample['Customer_Age']) - 1

        # encode Card Category feature with order
        sample['Card_Category'] = sample[['Card_Category']].replace(
            ['Blue', 'Silver', 'Gold', 'Platinum'], [0, 1, 2, 3])

        # encode categorical features
        need_impute = ['Education_Level', 'Marital_Status', 'Income_Category']
        for feature in need_impute:
            if all(sample[feature] == 'Unknown'):
                sample[feature] = np.nan
            else:
                sample[feature] = self.le[feature].transform(sample[feature])[
                    0]

        # impute all (in our case only categorical) features
        sample = pd.DataFrame(self.knni.transform(
            sample), columns=sample.columns)
        # after imputing categorical features we get float values, change and round it to integer
        sample[need_impute] = sample[need_impute].astype(np.int32)

        # drop unnecessary features
        sample = sample.drop(['Target'], axis=1)

        # set int type to integer features
        int_type = ['Customer_Age', 'Dependent_count', 'Card_Category', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Target', 'IsFemale']

        # scale float features
        float_type = ['Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                      'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        sample[float_type] = self.standard_scaler.transform(sample[float_type])

        return sample
