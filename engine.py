# Import necessary libraries and modules
from os import lseek
from ml_pipeline import utils
import configparser
import yaml
from ml_pipeline.data_analysis import DataAnalysis
from ml_pipeline.train import Train
from ml_pipeline.test import Test

# Define the main function
def main():
    # Read configuration from a YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config['mongo_host'])

    # Load data either from CSV files or a database based on the configuration
    # train_data, y_true, test_data = utils.data_loader_files()  # Load from CSV files
    train_data, y_true, test_data = utils.data_loader_database(config['mongo_host'])  # Load from a database

    # Preprocess the 'TEXT' column by applying the clean_text function
    train_data['TEXT'] = train_data['TEXT'].apply(utils.clean_text)

    # Initialize a DataAnalysis object for analyzing data
    data_analysis = DataAnalysis(config['figures_path'], train_data)

    # Perform frequency analysis on various columns
    data_analysis.frequency_analysis(train_data['Variation'], 'Variation')
    data_analysis.frequency_analysis(train_data['Gene'], 'Gene')
    data_analysis.frequency_analysis(y_true, 'Class')
    print("Data Analysis done!")

    # Initialize a Train object for training the machine learning model
    train = Train()

    # Split the data into training, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = train.split_data(train_data, y_true)
    print("Train Val Test split done!")

    # Check the distribution of data in all classes
    data_analysis.distribution_analysis(y_train, y_val, y_test)

    # Extract features from categorical data and text data
    train_variation_onehotCoding, val_variation_onehotCoding, test_variation_onehotCoding = train.extract_categorical_feature(X_train['Variation'], X_val['Variation'], X_test['Variation'])
    train_gene_onehotCoding, val_gene_onehotCoding, test_gene_onehotCoding = train.extract_categorical_feature(X_train['Gene'], X_val['Gene'], X_test['Gene'])
    train_text_feature, val_text_feature, test_text_feature = train.extract_text_feature(X_train['TEXT'], X_val['TEXT'], X_test['TEXT'], config['min_df_value'])
    print("Feature Extraction done!")

    # Concatenate extracted features
    train_all_features = utils.concatenate_features(train_gene_onehotCoding, train_variation_onehotCoding, train_text_feature)
    val_all_features = utils.concatenate_features(val_gene_onehotCoding, val_variation_onehotCoding, val_text_feature)
    test_all_features = utils.concatenate_features(test_gene_onehotCoding, test_variation_onehotCoding, test_text_feature)
    print("Features concatenated!")

    # Train a machine learning model and test it on unseen data
    train.train_model(train_all_features, y_train, val_all_features, y_val, test_all_features, y_test)
    train.test_unseen(test_data, train_variation_onehotCoding, train_gene_onehotCoding)

    # Print the length of the training data
    print(len(X_train))

    print("End")

# Run the main function
if __name__ == "__main__":
    main()
