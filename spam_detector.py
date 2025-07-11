# spam_detector.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import pickle
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Load dataset
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Drop unnecessary columns
df = df.iloc[:, :2]
df.columns = ['target', 'text']

# Encode target
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])

# EDA - Optional Visualizations

# Feature Engineering
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Preprocessing
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

df['transformed_text'] = df['text'].apply(transform_text)

# WordCloud (Optional)
# spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
# spam_wc.generate(" ".join(df[df['target']==1]['transformed_text']))
# plt.imshow(spam_wc)

# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual Models
mnb = MultinomialNB()
svc = SVC(probability=True)
etc = ExtraTreesClassifier()

# Train Voting Classifier
voting = VotingClassifier(estimators=[('NB', mnb), ('SVM', svc), ('ETC', etc)], voting='soft')
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)

# Stacking Classifier
stacking = StackingClassifier(estimators=[('NB', mnb), ('SVM', svc), ('ETC', etc)], final_estimator=RandomForestClassifier())
stacking.fit(X_train, y_train)
y_pred_stacking = stacking.predict(X_test)

# Evaluation
print("Voting Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Voting Precision:", precision_score(y_test, y_pred_voting, zero_division=0))
print("Voting Confusion Matrix:\n", confusion_matrix(y_test, y_pred_voting))

print("\nStacking Accuracy:", accuracy_score(y_test, y_pred_stacking))
print("Stacking Precision:", precision_score(y_test, y_pred_stacking, zero_division=0))
print("Stacking Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stacking))

# Serialize best model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
