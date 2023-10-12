# Import necessary libraries
import pandas as pd
from pymongo import MongoClient
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define a function to load data from a database
def data_loader_database(hostpath):
    """ Read the data from the database and organize it. """
    # Establish a connection to the MongoDB host
    conn = MongoClient(hostpath)

    # Select the database named 'CancerDB'
    db = conn.CancerDB

    # List all collections in the database
    print(db.list_collection_names())

    # Load data from 'train_variants' and 'train_text' collections
    training_variants = pd.DataFrame(list(db['train_variants'].find()))
    training_text = pd.DataFrame(list(db['train_text'].find()))

    # Merge two dataframes based on 'ID' and drop '_id_x' and '_id_y'
    train_data = pd.merge(training_text, training_variants, on="ID", how="left")
    train_data.drop(['_id_x', '_id_y'], axis=1, inplace=True)

    # Fill NA values in the 'TEXT' column
    train_data.loc[train_data['TEXT'].isnull(), 'TEXT'] = train_data['Gene'] + ' ' + train_data['Variation']

    # Convert the 'Class' column to integer and store it as 'y_true'
    y_true = train_data['Class'].astype(int)
    del train_data['Class']

    # Create the test set
    testing_variants = pd.DataFrame(list(db['test_variants'].find()))
    testing_text = pd.DataFrame(list(db['test_text'].find()))
    test_data = pd.merge(testing_text, testing_variants, on="ID", how="left")
    test_data.drop(['_id_x', '_id_y'], axis=1, inplace=True)

    # Fill NA values in the 'TEXT' column
    test_data.loc[test_data['TEXT'].isnull(), 'TEXT'] = test_data['Gene'] + ' ' + test_data['Variation']
    test_data.dropna(inplace=True)

    return train_data, y_true, test_data

# Define a function to load data from files
def data_loader_files():
    """ Read the data from files and organize it. """
    # Load text data from CSV files
    training_text = pd.read_csv('data/training_text.csv', sep="\|\|", engine="python", names=["ID", "TEXT"], skiprows=1)
    testing_text = pd.read_csv('data/test_text.csv', sep="\|\|", engine="python", names=["ID", "TEXT"], skiprows=1)

    # Load variant data from CSV files
    training_variants = pd.read_csv('data/training_variants.csv')
    testing_variants = pd.read_csv('data/test_variants.csv')

    # Merge two dataframes based on 'ID'
    train_data = pd.merge(training_text, training_variants, on="ID", how="left")

    # Fill NA values in the 'TEXT' column
    train_data.loc[train_data['TEXT'].isnull(), 'TEXT'] = train_data['Gene'] + ' ' + train_data['Variation']

    # Convert the 'Class' column to integer and store it as 'y_true'
    y_true = train_data['Class'].astype(int)
    del train_data['Class']

    # Create the test set
    test_data = pd.merge(testing_text, testing_variants, on="ID", how="left")

    # Fill NA values in the 'TEXT' column
    test_data.loc[test_data['TEXT'].isnull(), 'TEXT'] = test_data['Gene'] + ' ' + test_data['Variation']
    test_data.dropna(inplace=True)

    return train_data, y_true, test_data

# Define a function to clean and preprocess text
def clean_text(text):
    '''Preprocess the text by following some cleaning steps'''
    # Regular expressions for cleaning
    REPLACE_BY_SPACE_RE = re.compile('[/(){}[]|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^a-z]')

    # Initialize Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords
    STOPWORDS = set(stopwords.words('english'))

    # Convert text to lowercase
    text = text.lower()

    # Remove single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)

    # Apply cleaning operations
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)

    # Remove stopwords and apply Lemmatization
    cleaned_text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS)

    return cleaned_text

# Define a function to concatenate features
def concatenate_features(gene_feature, variation_feature, text_feature):
    '''Concatenate all extracted features together'''
    gene_variation_feature = pd.concat([variation_feature, gene_feature], axis=1)
    text_feature = pd.DataFrame(text_feature.toarray())
    gene_variation_feature.reset_index(drop=True, inplace=True)
    gene_variation_text_feature = pd.concat([text_feature, gene_variation_feature], axis=1)
    return gene_variation_text_feature
