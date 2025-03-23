import streamlit as st
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import tempfile
import subprocess

def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from a video file using FFmpeg."""
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the ChromaDB client and collection
client = chromadb.PersistentClient(path="/content")
try:
    collection = client.get_collection("Video_Subtitle_Search")
except ValueError:
    collection = client.create_collection("Video_Subtitle_Search")

def clean_text(text):
    """Cleans text by removing non-alphanumeric characters and stopwords, and applying lemmatization."""
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(clean_tokens).strip()

def extract_audio_text(audio_file):
    """Extracts text from an uploaded audio or video file."""
    recognizer = sr.Recognizer()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        if audio_file.type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(audio_file.read())
                temp_video_path = temp_video.name
            
            extract_audio_from_video(temp_video_path, temp_audio.name)
        else:
            temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error with the speech recognition service."

def get_most_relevant_subtitles(query):
    """Fetches the most relevant subtitles for the given query using cosine similarity."""
    cleaned_query = clean_text(query)
    query_embedding = model.encode([cleaned_query])
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=['documents', 'metadatas']
    )
    
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])
    
    relevant_subtitles = []
    if documents:
        for doc, meta in zip(documents[0], metadatas[0]):
            relevant_subtitles.append({'subtitle': doc, 'metadata': meta})
    
    return relevant_subtitles

# Streamlit UI
st.title("Enhanced Search Engine for Video Subtitles")
st.write("Upload an audio or video file to find relevant subtitles:")

uploaded_file = st.file_uploader("Upload Audio or Video File", type=["mp3", "wav", "mp4"])

if uploaded_file is not None:
    st.write("Processing file...")
    query = extract_audio_text(uploaded_file)
    st.write("Extracted Text:", query)
    
    if query:
        results = get_most_relevant_subtitles(query)
        
        if results:
            st.subheader("Search Results:")
            for res in results:
                st.markdown(f"**Subtitle:** {res['subtitle']}")
                st.markdown(f"**Metadata:** {res['metadata']}")
                st.write("---")
        else:
            st.write("No relevant subtitles found.")
