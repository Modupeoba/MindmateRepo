import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Mental_Health_FAQ.csv')

data.drop ('Question_ID', axis = 1, inplace = True)


def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Questions'].apply(preprocess_text)

y = data['tokenized Questions'].to_list()

tfidata_vectorizer = TfidfVectorizer()
corpus = tfidata_vectorizer.fit_transform(y)


st.markdown("<h1 style = 'color:  #451952; text-align: center; font-family:Garamond '>MindMate: Your Mental Wellness Chat Bot</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #673F69; text-align: center; font-family:Bradley Hand'> Built by OBAMUYI M.V. </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

hist_list = []

chat_response, robot_image = st.columns(2)
with robot_image:
    robot_image.image('chat-bot-concept-illustration_114360-5412-removebg-preview.png',width = 500, caption = 'I reply all your questions')

with chat_response:
    user_word = chat_response.text_input('Hello there you can ask your questions: ')
    def get_response(user_input):
        user_input_processed = preprocess_text(user_input) 

        user_input_vector = tfidata_vectorizer.transform([user_input_processed])

        similarity_scores = cosine_similarity(user_input_vector, corpus) 

        most_similar_index = similarity_scores.argmax() 

        return data['Answers'].iloc[most_similar_index] 

    greetings = ["Hey There....... How can I help",
                "Hi Human.... How can I help",
                'Twale baba nla, wetin dey happen nah',
                'How far Alaye, wetin happen'
                "Good Day .... How can I help", 
                "hello how are you doing today",
                "Hello There... How can I be useful to you today",
                "Hi Human.... How can I be of use"]

    exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
    farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

    random_farewell = random.choice(farewell) 
    random_greetings = random.choice(greetings) 


    if user_word.lower() in exits:
        chat_response.write(f"\nChatbot: {random_farewell}!")

    elif user_word.lower() in ['hi', 'hello', 'hey', 'hi there']:
        chat_response.write(f"\nChatbot: {random_greetings}!")

    elif user_word == '':
        chat_response.write('')
        
    else:   
        response = get_response(user_word)
        chat_response.write(f"\nChatbot: {response}")

        hist_list.append(user_word)


data_hist = None
with open('history.txt', 'a') as file:
    for item in hist_list:
        file.write(str(item) + '\n')
        file.write(response)

import csv
files = 'history.txt'
try:
    with open(files, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data_hist = list(reader)
except UnicodeDecodeError:
    with open(files, encoding='latin-1') as f:
        reader = csv.reader(f)
        data_hist = list(reader)
        
history = pd.DataFrame(data_hist)
st.sidebar.subheader('Chat History', divider=True)
st.sidebar.write(history)

#history = pd.Series(data)
#st.sidebar.subheader('Chat History', divider = True)
#st.sidebar.write(history)


st.header('Project Background Information',divider = True)
st.write("In response to the increasing prevalence of mental health challenges, we have developed a compassionate chatbot named Mind Matters. This chatbot aims to provide accessible and personalized support for individuals navigating their mental well-being, offering a safe space to express feelings, access resources and receive guidance")
