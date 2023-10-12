from xml.sax.handler import all_features
from ml_pipeline import utils  # Import custom utility functions
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class Test:
    def test(self, testdata, model, vectorizer, variation_feature_model, gene_feature_model):
        '''Test unseen data'''

        # Preprocess the text data in the 'TEXT' column
        testdata['TEXT'] = testdata['TEXT'].apply(utils.clean_text)

        # Extract features for Variation, Gene, and Text
        test_variation_onehotCoding = pd.get_dummies(testdata['Variation'], drop_first=True)
        test_variation_onehotCoding = test_variation_onehotCoding.reindex(columns=variation_feature_model.columns, fill_value=0)

        test_gene_onehotCoding = pd.get_dummies(testdata['Gene'], drop_first=True)
        test_gene_onehotCoding = test_gene_onehotCoding.reindex(columns=gene_feature_model.columns, fill_value=0)

        test_text_feature_onehotCoding = vectorizer.transform(testdata['TEXT'])
        test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

        # Combine all the extracted features
        all_features = utils.concatenate_features(test_variation_onehotCoding, test_gene_onehotCoding, test_text_feature_onehotCoding)

        # Use the pre-trained model to predict classes
        pred_prob = model.predict(all_features)
        print('Predicted classes of the 1st 10 samples of the unseen data')
        print(pred_prob[:10])
