# Genetic Mutations Multi-Class Classification

## Project Overview
Cancer tumors can have thousands of genetic mutations, making it challenging to distinguish driver mutations that contribute to tumor growth from passenger mutations. Currently, clinical pathologists manually interpret genetic mutations based on text-based clinical literature, which is a time-consuming process. Machine learning offers a solution to expedite this task with high accuracy.

In this project, we create features from medical literature data and develop a machine learning algorithm to automatically classify genetic variations. We experiment with different models, including Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Naive Bayes, to find the most suitable model.

---

## Aim
The primary objective of this project is to classify genetic mutations into nine classes based on medical literature.

---

## Data Description
The dataset is divided into variants and text for training and test datasets. It includes the following features:
- **ID:** The row identifier linking mutations to clinical evidence.
- **Gene:** The gene where the genetic mutation is located.
- **Variation:** The amino acid change for these mutations.
- **Class:** A numerical classification of the genetic mutation (1-9).
- **Text:** Clinical evidence used to classify the genetic mutation.

> Note: files >100 mb are excluded

---

## Tech Stack
- **Language:** Python
- **Libraries:** pymongo[srv] (used for database connectivity)

---

## Approach
- **Data Reading**
- **Data Analysis:** Utilizing libraries such as pandas, numpy, pretty_confusion_matrix, matplotlib, and sklearn to analyze data related to Class, Gene, and Variation.
- **Text Preprocessing**
- **Splitting Data, Evaluation, and Features Extraction**
- **Model Building:** Utilizing models like Logistic Regression, Random Forest, KNN, and Naive Bayes.
- **Hyperparameter Tuning:** Focusing on hyperparameter tuning for Logistic Regression.
- **Model Evaluation:** Including confusion matrix and log loss evaluation.

## Modular Code Overview
1. **lib:** A reference folder containing the original iPython notebook.
2. **ml_pipeline:** A folder with functions organized into different Python files, called by the `engine.py` script to run the project steps and train the model.
3. **requirements.txt:** Lists all required libraries and their versions. Install them with `pip install -r requirements.txt`.
4. **config.yaml:** Contains project specifications.
5. **readme.md:** Detailed instructions for running the code.

---

## Concepts explored

1. Text preprocessing steps, including lemmatization, tokenization, and using "Tfidf Vectorizer" for word relationships
2. Data splitting into training, testing, and validation sets
3. Multi-class classification
4. Evaluation metrics such as Log Loss and Confusion Matrix
5. Implementations of machine learning models, including Logistic Regression, KNN, Random Forest, and Naive Bayes.

---

