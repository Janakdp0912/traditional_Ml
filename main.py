import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Preprocess data (same steps as in your code)
df1 = df.drop(columns=['flags','category','response'], inplace=False)
columns_to_lower = ['instruction', 'intent']
df1[columns_to_lower] = df1[columns_to_lower].apply(lambda x: x.str.lower())
df1.columns = df1.columns.str.lower()
df2 = df1.drop_duplicates()

df2['label'] = pd.factorize(df2['intent'])[0]
categories = df2['intent'].unique()

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df2[['instruction']], df2['label'])
df2_resampled = pd.DataFrame({'instruction': X_resampled['instruction'], 'label': y_resampled})

X_train, X_test, y_train, y_test = train_test_split(
    df2_resampled['instruction'], df2_resampled['label'], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the models
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Define prediction functions
def predict_naive_bayes(instruction):
    instruction_tfidf = tfidf.transform([instruction])
    nb_pred = nb_model.predict(instruction_tfidf)
    return categories[nb_pred[0]]

def predict_svm(instruction):
    instruction_tfidf = tfidf.transform([instruction])
    svm_pred = svm_model.predict(instruction_tfidf)
    return categories[svm_pred[0]]

def predict_rf(instruction):
    instruction_tfidf = tfidf.transform([instruction])
    rf_pred = rf_model.predict(instruction_tfidf)
    return categories[rf_pred[0]]

# Streamlit interface
st.title("Customer Support Instruction Classification")

st.write("Enter the instruction below to classify it into one of the support categories:")

instruction = st.text_area("Instruction")

if instruction:
    # Naive Bayes Prediction
    nb_prediction = predict_naive_bayes(instruction)
    st.write(f"Naive Bayes Prediction: {nb_prediction}")

    # SVM Prediction
    svm_prediction = predict_svm(instruction)
    st.write(f"SVM Prediction: {svm_prediction}")

    # Random Forest Prediction
    rf_prediction = predict_rf(instruction)
    st.write(f"Random Forest Prediction: {rf_prediction}")
