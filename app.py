import streamlit as st
import speech_recognition as sr
import random
import torch
import os
from transformers import pipeline

# Set environment variables to avoid parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force PyTorch to use CPU instead of MPS (fix for macOS users)
torch.device("cpu")

# Load sentiment analysis model explicitly
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Define task recommendations based on mood
mood_tasks = {
    "POSITIVE": ["Creative brainstorming", "Team collaboration", "Innovative problem-solving"],
    "NEGATIVE": ["Relaxation exercises", "Mindfulness sessions", "Low-pressure tasks"],
    "NEUTRAL": ["Routine tasks", "Data entry", "Documentation"]
}

def get_mood(text):
    """Analyze sentiment and return the mood."""
    result = sentiment_analyzer(text)[0]
    return result["label"].upper()

def get_task(mood):
    """Return a random task based on mood."""
    return random.choice(mood_tasks.get(mood, ["General task"]))

def speech_to_text():
    """Convert speech to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)  # Added timeout to prevent indefinite listening
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio. Please try again."
        except sr.RequestError:
            return "Speech recognition service error. Try again later."
        except Exception as e:
            return f"Error: {str(e)}"

# Streamlit UI
st.title("🎯 AI-Powered Task Optimizer")

# Text Input
user_text = st.text_area("📝 Enter your mood description or use voice input:")

# Voice Input Button
if st.button("🎤 Use Voice Input"):
    user_text = speech_to_text()
    st.write("**You said:**", user_text)

# Analyze mood and recommend tasks
if user_text.strip():
    mood = get_mood(user_text)
    recommended_task = get_task(mood)
    
    st.subheader(f"🧠 Detected Mood: {mood}")
    st.success(f"✅ Recommended Task: {recommended_task}")
