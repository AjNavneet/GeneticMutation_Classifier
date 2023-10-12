from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.linear_model import LogisticRegression
from ml_pipeline.test import Test
import cv2
import pandas as pd

class Train:
    def evaluate_model(self, model, test_features, y_truth, datatype):
        '''Evaluate different models using confusion matrix and log loss'''
        pred = model.predict(test_features)
        how = pp_matrix_from_data(y_truth, pred)
        pred_prob = model.predict_proba(test_features)
        eval = log_loss(y_truth, pred_prob)
        print("Log Loss for " + datatype + " data")
        print(eval)

    def train_model(self, train_all_features, y_train, val_all_features, y_val, test_all_features, y_test):
        '''Train a logistic regression model and evaluate it on different datasets'''
        loj = LogisticRegression()
        self.loj_model = loj.fit(train_all_features, y_train)
        self.evaluate_model(self.loj_model, train_all_features, y_train, 'training')
        self.evaluate_model(self.loj_model, val_all_features, y_val, 'validation')
        self.evaluate_model(self.loj_model, test_all_features, y_test, 'testing')

    def test_unseen(self, test_data, variation_feature_model, gene_feature_model):
        '''Predict the data without labels using a pre-trained model'''
        test = Test()
        test.test(test_data, self.loj_model, self.text_vectorizer, variation_feature_model, gene_feature_model)

    def split_data(self, train_data, y_true):
        '''Split the data into test, validation, and train while maintaining the same distribution of output variables'''
        X_train, X_rem, y_train, y_rem = train_test_split(train_data, y_true, stratify=y_true, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.5)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def extract_categorial_feature(self, X_train, X_val, X_test):
        '''Extract categorical features using one-hot encoding'''
        train_feature_onehotCoding = pd.get_dummies(X_train, drop_first=True)
        val_feature_onehotCoding = pd.get_dummies(X_val, drop_first=True)
        val_feature_onehotCoding = val_feature_onehotCoding.reindex(columns=train_feature_onehotCoding.columns, fill_value=0)
        test_feature_onehotCoding = pd.get_dummies(X_test, drop_first=True)
        test_feature_onehotCoding = test_feature_onehotCoding.reindex(columns=train_feature_onehotCoding.columns, fill_value=0)
        return train_feature_onehotCoding, val_feature_onehotCoding, test_feature_onehotCoding

    def extract_text_feature(self, X_train, X_val, X_test, min_df_value):
        '''Extract text features using TF-IDF method'''
        self.text_vectorizer = TfidfVectorizer(min_df=min_df_value, stop_words="english")
        train_text_feature_onehotCoding = self.text_vectorizer.fit_transform(X_train)
        val_text_feature_onehotCoding = self.text_vectorizer.transform(X_val)
        test_text_feature_onehotCoding = self.text_vectorizer.transform(X_test)
        train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)
        val_text_feature_onehotCoding = normalize(val_text_feature_onehotCoding, axis=0)
        test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)
        return train_text_feature_onehotCoding, val_text_feature_onehotCoding, test_text_feature_onehotCoding
