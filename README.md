---

# ARTICULATE: A Multipurpose AI Speech Companion

Articulate is an AI-powered speech companion designed to assist users in improving their communication skills, emotional well-being, and daily productivity. This project integrates advanced technologies such as speech recognition, emotion detection, and real-time feedback systems to create a comprehensive tool for enhancing speech clarity, fluency, and confidence.


![image](https://github.com/user-attachments/assets/cd142782-a820-4c05-980d-da1bbebaa78a)




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

![image](https://github.com/user-attachments/assets/e07bf813-f82f-42dc-b8c9-e5303485892c)



### Sentiment Analysis
1. **Speech-to-Text Conversion**: Converts audio to text using the Google Web Speech API.
2. **Text Analysis**: Employs TextBlob for polarity analysis, classifying sentiment as Positive, Negative, or Neutral.

![image](https://github.com/user-attachments/assets/b4107247-7497-4015-bc00-36de91873ed6)



### Sign Language Integration
- Converts speech to sign language using on-screen avatars, aiding individuals with hearing impairments.
selecting images of Sign language for further prediction 
![image](https://github.com/user-attachments/assets/b5bda192-8847-416c-bdb3-e785cf60ccac)



predicted text based on the Sign language and its corresponding speech 
![image](https://github.com/user-attachments/assets/9b068191-aea9-42b6-b7ea-5d86333b970c)




real time prediction of Sign Language 
![image](https://github.com/user-attachments/assets/d608a598-82b6-47b6-a1ef-3607ee90985b)




### Filler Removal
- Detects and removes unnecessary fillers (e.g., "um," "uh") from speech to enhance clarity.

![image](https://github.com/user-attachments/assets/4aa611f0-b96f-45cd-a678-71dc2de32d93)


### Audio Handling and Conversion
- Handles audio files in various formats, converting non-WAV files to WAV using PyDub and FFmpeg.

## Contributors
This project was collaboratively developed by:
- **Joshwin Isac**
- **Sai Darshan**
- **Suhas S**

---


Below Given is the Video demonstrating the project Articulate :
https://drive.google.com/file/d/1SlGRdXSyB3Pc_zW5AZonZ7K-WsDU1aSF/view





