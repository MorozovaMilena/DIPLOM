from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
import gensim
from gensim import corpora
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import webbrowser
import threading


nltk.download('stopwords')

app = Flask(__name__)


def custom_load_model(filepath, custom_objects=None, compile=False):
    from keras.models import load_model
    custom_objects = custom_objects or {}
    custom_objects['Adam'] = tf.keras.optimizers.Adam
    model = load_model(filepath, custom_objects=custom_objects, compile=compile)
    return model

model = custom_load_model('combined_model.h5')


lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
dictionary = corpora.Dictionary.load('dictionary')


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


contractions_dict = { 
    "ain't": "are not","'s":" is","aren't": "are not",
    "can't": "cannot","can't've": "cannot have",
    "'cause": "because","could've": "could have","couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
    "don't": "do not","hadn't": "had not","hadn't've": "had not have",
    "hasn't": "has not","haven't": "have not","he'd": "he would",
    "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
    "how'd": "how did","how'd'y": "how do you","how'll": "how will",
    "I'd": "I would", "I'd've": "I would have","I'll": "I will",
    "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
    "it'd": "it would","it'd've": "it would have","it'll": "it will",
    "it'll've": "it will have", "let's": "let us","ma'am": "madam",
    "mayn't": "may not","might've": "might have","mightn't": "might not", 
    "mightn't've": "might not have","must've": "must have","mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
    "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
    "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have","should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
    "that'd": "that would","that'd've": "that would have", "there'd": "there would",
    "there'd've": "there would have", "they'd": "they would",
    "they'd've": "they would have","they'll": "they will",
    "they'll've": "they will have", "they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we would",
    "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
    "what'll've": "what will have","what're": "what are", "what've": "what have",
    "when've": "when have","where'd": "where did", "where've": "where have",
    "who'll": "who will","who'll've": "who will have","who've": "who have",
    "why've": "why have","will've": "will have","won't": "will not",
    "won't've": "will not have", "would've": "would have","wouldn't": "would not",
    "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have","y'all're": "you all are",
    "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
    "you'll": "you will","you'll've": "you will have", "you're": "you are",
    "you've": "you have"}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def preprocess_text(text):
    text = expand_contractions(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_and_tokenize(text, tokenizer, max_length=100):
    text = preprocess_text(text)  # Apply the same preprocessing
    tokens = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
    )
    return tokens

def get_lda_features_single(text, lda_model, dictionary):
    corpus = [dictionary.doc2bow(text.split())]
    topics = lda_model[corpus]
    lda_features = np.zeros((1, lda_model.num_topics))
    for i, doc in enumerate(topics):
        for topic_num, prob in doc[0]:
            lda_features[i, topic_num] = prob
    return lda_features

def concatenate_features(bert_features, lda_features):
    concatenated_features = np.concatenate([bert_features, lda_features], axis=1)
    return concatenated_features



def predict_news_with_probability(text):
    new_tokens = preprocess_and_tokenize(text, bert_tokenizer)
    new_lda_features = get_lda_features_single(text, lda_model, dictionary)
    new_bert_features = bert_model([new_tokens['input_ids'], new_tokens['attention_mask']])[0]
    new_bert_features_cls = new_bert_features[:, 0, :]  # Get [CLS] token features
    new_data_features = concatenate_features(new_bert_features_cls, new_lda_features)
    
    prediction_prob = model.predict(new_data_features)[0][0]
    
    
    if prediction_prob > 0.4:
        result = "Fake News"
    else:
        result = "Real News"
    
    return result, prediction_prob

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['news']
    result, prediction_prob = predict_news_with_probability(data)
    return render_template('index.html', prediction_text=f'{result}', news_text=data)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)