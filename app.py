from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)
model = load_model('models/lstm.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global tokenizer
    # Get uploaded file from request
    uploaded_file = request.files['file']
    
    # Preprocess the uploaded audio file if necessary
    # For simplicity, let's assume you preprocess the audio into text data
    # and then use the LSTM model
    
    # Convert audio to text (replace this with actual audio preprocessing)
    audio_text = preprocess_audio(uploaded_file)
    
    # Tokenize the text
    tokenizer.fit_on_texts([audio_text])
    sequences = tokenizer.texts_to_sequences([audio_text])
    
    # Pad sequences to fixed length
    sequences = sequence.pad_sequences(sequences, maxlen=maxlen)
    
    # Make predictions using the LSTM model
    prediction = model.predict(sequences)
    
    # Assuming it's a binary classification task, convert prediction to class label
    predicted_class = "Happy" if prediction > 0.5 else "Sad"
    
    # Return prediction results
    return f"Prediction: {predicted_class}, Probability: {prediction[0][0]:.4f}"

def preprocess_audio(audio_file):
    # Placeholder function for audio preprocessing
    # You need to implement the actual audio preprocessing here
    # For example, you can use libraries like librosa to extract features
    # from the audio file (e.g., spectrograms or MFCC features)
    # For simplicity, let's assume this function converts audio to text
    return "This is an example audio text."

if __name__ == '__main__':
    app.run(debug=True)