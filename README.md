![image](https://github.com/user-attachments/assets/cb9151a3-bdec-434a-91da-9332fe94dfd5)Here’s a well-structured README file for your GitHub repository:

---

# ARTICULATE: A Multipurpose AI Speech Companion

Articulate is an AI-powered speech companion designed to assist users in improving their communication skills, emotional well-being, and daily productivity. This project integrates advanced technologies such as speech recognition, emotion detection, and real-time feedback systems to create a comprehensive tool for enhancing speech clarity, fluency, and confidence.
![image](https://github.com/user-attachments/assets/bf7caecc-9c9c-48e4-a6ac-204e65134c40)



## Features
- **Speech Clarity and Fluency Improvement**: Offers personalized guidance and corrective feedback to empower users to communicate more effectively.
- **Voice-Driven Task Manager**: Enables users to organize daily tasks using voice commands, with NLP systems converting instructions into actionable tasks. Future integration with productivity apps is planned.
- **Accessibility for Individuals with Disabilities**:
  - Translation of speech into sign language through on-screen avatars.
  - Real-time speech-to-text conversion for seamless communication in diverse settings.
- **Speech Style Feedback**: Analyzes pitch and tone to provide suggestions for improving speaking abilities.
- **Filler Removal**: Automatically detects and removes fillers from speech to enhance communication quality.

## Technologies and Tools Used
### Web Framework
- **Flask**: For creating a web-based interface to upload audio and display results.

### Python Libraries
- **Librosa**: For audio feature extraction.
- **SoundFile**: For reading and processing audio files.
- **NumPy**: For numerical computations.
- **SpeechRecognition**: For speech-to-text conversion using the Google Web Speech API.
- **TextBlob**: For sentiment analysis of text.
- **PyDub**: For audio format conversion.

### Machine Learning
- **MLP Classifier**: A trained Multi-Layer Perceptron model for emotion classification with 83% accuracy.
- **PCA (Principal Component Analysis)**: For dimensionality reduction of extracted features.

### Utilities
- **UUID**: For generating unique filenames.
- **FFmpeg**: For handling audio format conversions.

## Proposed Methodology
### Emotion Prediction
1. **Feature Extraction**: Extracts features such as MFCC (Mel Frequency Cepstral Coefficients), chroma features, and Mel spectrogram using Librosa.
2. **Dimensionality Reduction**: Applies Principal Component Analysis (PCA) to reduce feature dimensions before classification.
3. **Classification**: Utilizes a pre-trained Multi-Layer Perceptron (MLP) model to predict emotions with an accuracy of 83%.

![image](https://github.com/user-attachments/assets/493d6195-8457-4e68-9a53-f1780d1bf654)


### Sentiment Analysis
1. **Speech-to-Text Conversion**: Converts audio to text using the Google Web Speech API.
2. **Text Analysis**: Employs TextBlob for polarity analysis, classifying sentiment as Positive, Negative, or Neutral.

![image](https://github.com/user-attachments/assets/5ad6c5e8-203f-4867-9154-0d8145236374)


### Sign Language Integration
- Converts speech to sign language using on-screen avatars, aiding individuals with hearing impairments.
selecting images of Sign language for further prediction 
![image](https://github.com/user-attachments/assets/574da6a3-7c2e-4a0e-afd2-26363784ef4c)


predicted text based on the Sign language and its corresponding speech 
![image](https://github.com/user-attachments/assets/8f0b01ca-b90f-4c48-91d5-1296ad98796f)



real time prediction of Sign Language 
![Uploading image.png…]()



### Filler Removal
- Detects and removes unnecessary fillers (e.g., "um," "uh") from speech to enhance clarity.

![image](https://github.com/user-attachments/assets/ae61b044-d594-4bce-aeac-1cbe3ec7791a)


### Audio Handling and Conversion
- Handles audio files in various formats, converting non-WAV files to WAV using PyDub and FFmpeg.

## Contributors
This project was collaboratively developed by:
- **Joshwin Isac**
- **Sai Darshan**
- **Suhas S**

---
