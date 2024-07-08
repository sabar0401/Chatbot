import os
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, TFBertForSequenceClassification, TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import tensorflow as tf
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load Spacy model for NER
nlp = spacy.load('en_core_web_sm')

# Load pre-trained BERT model and tokenizer for intent recognition
intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
intent_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load pre-trained model for response generation (using BART in this example)
response_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
response_model = TFAutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

# Predefined intents (for demonstration purposes)
intents = {
    "greeting": ["hello", "hi", "hey"],
    "goodbye": ["bye", "see you", "goodbye"],
    "thanks": ["thank you", "thanks", "cheers"]
}

# Sample conversational data (for training purposes)
data = [
    {"text": "hello", "intent": "greeting"},
    {"text": "hi there", "intent": "greeting"},
    {"text": "bye", "intent": "goodbye"},
    {"text": "thank you", "intent": "thanks"}
]

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(filtered_tokens)

# Prepare data for intent recognition
def prepare_intent_data(data):
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['intent'] for item in data]
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    
    inputs = intent_tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
    return inputs, labels, label_encoder

# Train intent recognition model
def train_intent_model(data):
    inputs, labels, label_encoder = prepare_intent_data(data)
    X_train, X_test, y_train, y_test = train_test_split(inputs['input_ids'], labels, test_size=0.2)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    intent_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    intent_model.fit(X_train, y_train, epochs=3, batch_size=16)
    
    return label_encoder

label_encoder = train_intent_model(data)

# Recognize intent
def recognize_intent(text):
    inputs = intent_tokenizer([preprocess_text(text)], return_tensors='tf', padding=True, truncation=True)
    outputs = intent_model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
    return label_encoder.inverse_transform([predicted_label])[0]

# Extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Generate response
def generate_response(input_text):
    inputs = response_tokenizer([input_text], return_tensors='tf', max_length=512, truncation=True)
    outputs = response_model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    response = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main chatbot function
def chatbot(input_text):
    intent = recognize_intent(input_text)
    entities = extract_entities(input_text)
    response = generate_response(input_text)
    return {
        "intent": intent,
        "entities": entities,
        "response": response
    }

# Testing the chatbot
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        result = chatbot(user_input)
        print(f"Chatbot: {result['response']}")
        print(f"Detected Intent: {result['intent']}")
        print(f"Extracted Entities: {result['entities']}")