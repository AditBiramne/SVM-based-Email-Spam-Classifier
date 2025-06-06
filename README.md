# 📧 SVM Based Email Spam Classifier (Support vector Machine)

This project implements a machine learning model to classify emails as **spam** or **not spam (ham)** using a Support Vector Machine (SVM) classifier. It demonstrates data preprocessing, feature extraction, model training, and evaluation with Python’s scikit-learn library.

---

## 🧩 Project Overview

* Detect whether an email is spam or ham using text classification techniques  
* Use SVM, a powerful supervised learning algorithm for classification tasks  
* Perform text preprocessing and convert emails into numeric features using TF-IDF  
* Evaluate model performance with accuracy, precision, recall, and F1-score metrics  

---

## 📂 Dataset

The dataset consists of labeled email samples:

* Emails classified as **spam** or **ham**  
* Includes raw email text data  
* Typically sourced from public datasets like the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/) or UCI ML Repository  

---

## 🛠️ Tech Stack

* Python 3.x  
* Libraries:  
  * `pandas` – Data handling  
  * `numpy` – Numerical operations  
  * `scikit-learn` – Machine learning models and evaluation  
  * `nltk` – Natural Language Toolkit for text preprocessing (optional)  

---

## 🚀 How to Run

### Clone the repository

<pre><code>git clone https://github.com/yourusername/email-spam-classifier-svm.git
cd email-spam-classifier-svm
</code></pre>

### Install required libraries

<pre><code>pip install pandas numpy scikit-learn nltk
</code></pre>

### Run the training and evaluation script

<pre><code>python spam_classifier.py
</code></pre>

---

## 📋 Code Highlights

### Example: Loading Data and Preprocessing

<pre><code>import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('emails.csv')

# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['email_text'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# Train SVM classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred))
</code></pre>

---

## 📈 Evaluation Metrics

* **Accuracy:** Overall correctness of the model  
* **Precision:** Percentage of predicted spam emails that are actually spam  
* **Recall:** Percentage of actual spam emails correctly identified  
* **F1-Score:** Harmonic mean of precision and recall  

---

## 🔮 Future Enhancements

* Experiment with other ML algorithms like Random Forest or Gradient Boosting  
* Use word embeddings (e.g., Word2Vec, GloVe) for richer feature representation  
* Deploy the model as a web service or API  
* Incorporate email metadata features (sender, subject, etc.)  

---

*Happy coding and spam-free inboxes!* 🚀
