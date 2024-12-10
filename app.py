from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import PCA
from textblob import TextBlob
import speech_recognition as sr
from pydub import AudioSegment
import uuid

app = Flask(__name__)

#FFmpeg converter path for PyDub
AudioSegment.converter = "C:/Users/saida/OneDrive/Desktop/Projects/SPR_Project/ffmpeg/bin/ffmpeg.exe"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#trained model and PCA transformer
with open('best_mlp_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pca_transformer.pkl', 'rb') as f:
    pca = pickle.load(f)
    
# Function to extract features from a given file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])  # Initialize empty result array
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Function to preprocess input audio and predict the emotion
def predict_emotion(file_path):
    feature = extract_feature(file_path)
    feature_pca = pca.transform([feature])  # Use pre-fitted PCA for transformation
    prediction = model.predict(feature_pca)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

# Route to render the emotion page
@app.route('/emotion')
def emotion():
    return render_template('emotion.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/filler')
def filler():
    return render_template('filler.html')

# Route to handle file upload and prediction
@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    # Check if file is present in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio part in the request."}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file."}), 400

    # Generate a unique filename to prevent conflicts
    unique_id = uuid.uuid4().hex
    
    # Determine the file extension
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    input_filename = f"{unique_id}{file_ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.wav")
    
    try:
        # Save the uploaded file to the 'uploads' folder
        audio_file.save(input_path)
        
        # If the file is not already a WAV, convert it
        if file_ext != '.wav':
            try:
                audio = AudioSegment.from_file(input_path)
                audio.export(wav_path, format="wav")
                # Remove the original input file
                os.remove(input_path)
            except Exception as convert_error:
                # If conversion fails, try to use the original file directly
                print(f"Conversion error: {convert_error}")
                wav_path = input_path
        else:
            # If it's already a WAV, just rename the file
            wav_path = input_path
        
        # Predict emotion from the WAV file
        prediction = predict_emotion(wav_path)
        
        # Remove the temporary WAV file
        os.remove(wav_path)
        
        # Return the result as a JSON response
        return jsonify({"emotion": prediction})
    
    except Exception as e:
        # In case of any error, ensure that files are removed
        if os.path.exists(input_path):
            os.remove(input_path)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({"error": str(e)}), 500

# Function to convert speech to text
def convert_speech_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)  # Using Google Web Speech API
    return text

@app.route('/filter_filler', methods=['POST'])
def filter_filler():
    # Check if file is present in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio part in the request."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file."}), 400

    # Generate a unique filename to prevent conflicts
    unique_id = uuid.uuid4().hex
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    input_filename = f"{unique_id}{file_ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.wav")

    try:
        # Save the uploaded file to the 'uploads' folder
        audio_file.save(input_path)

        # If the file is not already a WAV, convert it
        if file_ext != '.wav':
            audio = AudioSegment.from_file(input_path)
            audio.export(wav_path, format="wav")
            os.remove(input_path)
        else:
            wav_path = input_path

        # Convert speech to text
        text = convert_speech_to_text(wav_path)

        # Filter filler words
        filtered_text = filter_filler_words(text)

        # Remove the temporary WAV file
        os.remove(wav_path)

        # Return the result as a JSON response
        return jsonify({"original_text": text, "filtered_text": filtered_text})

    except Exception as e:
        # Clean up files in case of errors
        if os.path.exists(input_path):
            os.remove(input_path)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({"error": str(e)}), 500


# Function to filter filler words from text
def filter_filler_words(text):
    filler_words = {"uh", "um", "like", "you know", "er", "ah", "hmm"}
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in filler_words]
    return " ".join(filtered_words)


# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    words = analysis.words
    highlighted_text = ""
    
    for word in words:
        word_analysis = TextBlob(word)
        polarity = word_analysis.sentiment.polarity  # Polarity ranges from -1 to 1
        if polarity > 0:
            highlighted_text += f'<span style="background-color: yellow; color: black;">{word}</span> '
        elif polarity < 0:
            highlighted_text += f'<span style="background-color: red; color: white;">{word}</span> '
        else:
            highlighted_text += f'{word} '
    
    overall_polarity = analysis.sentiment.polarity
    if overall_polarity > 0:
        sentiment = "Positive"
    elif overall_polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, highlighted_text.strip()


# Route to handle sentiment prediction
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    # Check if file is present in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio part in the request."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file."}), 400

    # Generate a unique filename to prevent conflicts
    unique_id = uuid.uuid4().hex
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    input_filename = f"{unique_id}{file_ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.wav")

    try:
        # Save the uploaded file to the 'uploads' folder
        audio_file.save(input_path)

        # If the file is not already a WAV, convert it
        if file_ext != '.wav':
            audio = AudioSegment.from_file(input_path)
            audio.export(wav_path, format="wav")
            os.remove(input_path)
        else:
            wav_path = input_path

        # Convert speech to text
        text = convert_speech_to_text(wav_path)

        # Analyze sentiment and highlight words
        sentiment, highlighted_text = analyze_sentiment(text)

        # Remove the temporary WAV file
        os.remove(wav_path)

        # Return the result as a JSON response
        return jsonify({"text": highlighted_text, "sentiment": sentiment})

    except Exception as e:
        # Clean up files in case of errors
        if os.path.exists(input_path):
            os.remove(input_path)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)