from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
CORS(app)  # Enable CORS to allow React frontend to communicate

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.wordpunct_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_sms = data.get('message', '')
    
    # Preprocess
    transformed_sms = transform_text(input_sms)
    
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict
    result = model.predict(vector_input)[0]
    
    # Return result
    return jsonify({
        'prediction': 'Spam Message' if result == 1 else 'Not a Spam Message'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)