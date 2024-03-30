from flask import Flask, request, render_template
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'speech2\models\lstm.h5'  
model = load_model(MODEL_PATH)

# Assuming the same labels and function for MFCC extraction
label_encoder = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            return render_template('upload.html', prediction="No file selected")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', prediction="No file selected")
        
        # If the user provides a file
        if file:
            # Save the file temporarily
            file_path = "temp.wav"
            file.save(file_path)
            
            # Extract MFCC features and predict sentiment
            mfcc_features = extract_mfcc(file_path)
            mfcc_features_reshaped = np.expand_dims(np.expand_dims(mfcc_features, axis=0), axis=-1)
            predictions = model.predict(mfcc_features_reshaped)
            predicted_sentiment = label_encoder[np.argmax(predictions)]
            
            return render_template('upload.html', prediction=predicted_sentiment)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
