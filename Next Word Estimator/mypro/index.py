from flask import *
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
from doc31 import training_doc3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
app = Flask(__name__)

@app.route("/")
@app.route("/res",methods=["POST","GET"])
def home():
    if request.method=="POST":
        print("hai")
        val=request.form['val']
        input_text = val.strip().lower()

        cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
        tokens = word_tokenize(cleaned)
        train_len = 3+1
        text_sequences = []
        
        for i in range(train_len,len(tokens)):
            seq = tokens[i-train_len:i]
            text_sequences.append(seq)
        sequences = {}
        count = 1
        

        for i in range(len(tokens)):
            if tokens[i] not in sequences:
                sequences[tokens[i]] = count
                count += 1

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_sequences)
        sequences = tokenizer.texts_to_sequences(text_sequences) 

        #Collecting some information   
        vocabulary_size = len(tokenizer.word_counts)+1

        n_sequences = np.empty([len(sequences),train_len], dtype='int32')
        for i in range(len(sequences)):
            n_sequences[i] = sequences[i]

        train_inputs = n_sequences[:,:-1]
        train_targets = n_sequences[:,-1]
        train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
        seq_len = train_inputs.shape[1]

        model = load_model("mymodel.h5")
        
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre') 
        print(encoded_text, pad_encoded)
        print("next word suggestion:")
        d=[]
        for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = tokenizer.index_word[i]
            pw=input_text+" "+pred_word
            d.append(pw)
        return render_template("index.html",que=input_text,data=d)  
    return render_template("index.html",data=None)


if __name__ == '__main__':
    app.run(debug=True)