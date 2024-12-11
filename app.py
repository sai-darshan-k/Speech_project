from flask import Flask, render_template, request, jsonify, send_from_directory
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from textblob import TextBlob
import speech_recognition as sr
from pydub import AudioSegment
import torch.nn as nn
from gtts import gTTS
import soundfile as sf
import pickle
import librosa
import numpy as np
import time
import uuid
import torch
import cv2
import os

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model and categories
model_path = "indian_sign_language_model.pth"
categories = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
num_classes = len(categories)

# Instantiate and load the model
# Sign Language CNN model
sign_language_model = SignLanguageCNN(num_classes=num_classes)
sign_language_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
sign_language_model.eval()

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = Flask(__name__)

#FFmpeg converter path for PyDub
AudioSegment.converter = "C:/Users/saida/OneDrive/Desktop/Projects/SPR_Project/ffmpeg/bin/ffmpeg.exe"
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Helper function to recognize a hand sign
def recognize_hand_sign(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = sign_language_model(img)
        _, predicted = torch.max(outputs, 1)
        return categories[predicted.item()]
    
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

@app.route("/sign")
def sign():
    return render_template("sign.html")

import time
import os

@app.route("/upload", methods=["POST"])
def upload_file():
    if "files" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist("files")
    recognized_chars = []

    # Process each file (i.e., hand sign image)
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Recognize the hand sign from the uploaded image
        recognized_char = recognize_hand_sign(file_path)
        recognized_chars.append(recognized_char)

    # Form the word from the recognized characters
    word = ''.join(recognized_chars)

    # Generate TTS (Text-to-Speech) output with dynamic filename
    timestamp = str(int(time.time()))  # Using timestamp for unique filename
    tts_path = os.path.join(AUDIO_FOLDER, f"{timestamp}_output.mp3")
    tts = gTTS(text=word, lang="en")
    tts.save(tts_path)

    # Generate the full path to the audio file
    audio_url = f"/static/audio/{os.path.basename(tts_path)}"

    return jsonify({
        "prediction": word,
        "audio": audio_url
    })

@app.route("/camera", methods=["POST"])
def camera_detection():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, "camera_capture.jpg")
    file.save(file_path)

    recognized_char = recognize_hand_sign(file_path)
    if recognized_char is None:
        return jsonify({"error": "Prediction failed"}), 500

    # Generate TTS (Text-to-Speech) output with dynamic filename
    timestamp = str(int(time.time()))  # Using timestamp for unique filename
    tts_path = os.path.join(AUDIO_FOLDER, f"{timestamp}_output.mp3")
    tts = gTTS(text=recognized_char, lang="en")
    tts.save(tts_path)

    return jsonify({"prediction": recognized_char, "audio": tts_path})

@app.route("/static/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

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
    filler_words = {"um", "uh", "like", "you know", "actually", "basically", "seriously", "I mean", "so", "right", "well", "literally", "kind of", "sort of", "err", "uhh", "okay"}
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