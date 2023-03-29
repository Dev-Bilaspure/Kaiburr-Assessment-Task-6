# Import necessary libraries
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize a Porter Stemmer instance
ps = PorterStemmer()

# Define a function for text preprocessing
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    y = []
    # Remove non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    # Return the preprocessed text
    return " ".join(y)

# Load the saved TF-IDF vectorizer and trained model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Create a Streamlit application titled "Spam Message Classifier"
st.title("Spam Message Classifier")

# Add a text area for user input
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed message using the saved vectorizer
    vector_input = tfidf.transform([transformed_sms])
    # Use the trained model to predict the label of the message
    result = model.predict(vector_input)[0]
    # Display the predicted label
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")