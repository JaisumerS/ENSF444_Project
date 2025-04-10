#%pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from ML_models import create_pipeline, inputters, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
print("Path to dataset files:", path)

# The code above is from the Kaggle website where it gives a sample of how to download

'''
Step 1 of ML workflow: Reading in Dataset

'''

# joins path above with directory created
base_path = os.path.join(path, "News _dataset")

# reads cvs files
fake_df = pd.read_csv(os.path.join(base_path, "Fake.csv"))
true_df = pd.read_csv(os.path.join(base_path, "True.csv"))

# checking whether they loaded in correctly
display(fake_df.head())
display(true_df.head())

'''
Step 2 of the ML Workflow: Data processing

'''

# Checking to see whether there are any missing values
print("Missing values in fake news file:")
print(fake_df.isnull().sum())

print("\nMissing values in true news file:")
print(true_df.isnull().sum())\

# Checking to see if the same datatype is in each column
true_types_df = true_df.applymap(type)
fake_types_df = fake_df.applymap(type)

true_type_counts = true_types_df.nunique()
fake_type_counts = fake_types_df.nunique()

print("\nUnique datatypes in true news file: \n", true_type_counts)
print("\nUnique datatypes in fake news file: \n", fake_type_counts)

# Adding labels to fake and true news data, where fake = 1 and true = 0
fake_df['label'] = 1
true_df['label'] = 0

# Concatenating the fake and true news data and shuffling it
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# Combining the title and text columns to create a single input for the vectorizer
df['combined_title_and_text'] = df['title'] + " " + df['text']
X = df['combined_title_and_text']
y = df['label']

# Splitting the data into a 75% train and 25% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Creating a pipeline with a TF-IDF transformer and an SVM (LinearSVC) classifier or one to admin choosing
model = input("<><><>For Admin<><><>\nChoose a model out of \n1.SVC Model\n2.Multinomial NB\n3.Random Forest, press the number of the model or click enter")
pipeline = create_pipeline(model)

# Fitting the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predicting labels for the test set
y_pred = pipeline.predict(X_test)


metricview = input("Would you like to see the metrics of the model used (WARNING: may take a minute)? Enter y or n for yes or no: ")
if metricview.low() == 'y':
    metrics(pipeline, X_train, y_train, X_test, y_test, y_pred)
elif metricview.low() == 'n':
    print("Be sure to comeback for Fact or Fiction News Checking!!!!")
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

fullarticle = inputters()