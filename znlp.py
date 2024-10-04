# hello_world/ZB/znlp.py
# This is Zara's Natural Language Processing (NLP) handler, supporting dynamic API fallbacks.

import json
import openai
import requests
from utils.logger import Logger
from openai import OpenAIError, RateLimitError, BadRequestError
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class ZbNlp:
    def __init__(self, config_path='hello_world/config.json'):
        """
        Initializes ZbNlp with API keys and logging. Supports multiple NLP services.
        :param config_path: Path to the config file containing API keys.
        """
        self.config = self.load_config(config_path)
        self.logger = Logger(log_file_path='logs/znlp.log')

        # API keys for multiple services
        self.openai_api_key = self.config.get("openai_api_key")
        self.huggingface_api_url = self.config.get("huggingface_api_url")
        
        # Initialize OpenAI API if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("OpenAI API initialized.")
        else:
            self.logger.error("OpenAI API key is missing.")
        
        # Initialize Hugging Face pipeline if available
        if self.huggingface_api_url:
            self.hugging_face_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.logger.info("Hugging Face sentiment-analysis initialized.")
        
        # Initialize Scikit-learn components for fallback tasks (classification)
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

        self.logger.info("Zira NLP initialized with available APIs.")

    def load_config(self, path):
        """
        Loads the configuration file and returns the API keys and other settings.
        :param path: Path to the configuration file.
        :return: Parsed configuration as a dictionary.
        """
        try:
            with open(path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            self.logger.error("Configuration file not found.")
            raise
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def analyze_text_openai(self, text, engine="davinci", max_tokens=100, temperature=0.7, n=1, stop=None, log_to_file=True):
        """
        Analyzes text using the OpenAI API. If OpenAI is unavailable, falls back to Hugging Face.
        :param text: The text to analyze.
        :param engine: The OpenAI engine to use (e.g., "davinci").
        :param max_tokens: Maximum number of tokens in the output.
        :param temperature: The randomness of the response.
        :param n: Number of completions to generate.
        :param stop: Sequences to stop at.
        :param log_to_file: Whether to log the result.
        :return: The result text.
        """
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=text,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                stop=stop
            )
            result = response.choices[0].text.strip()
            if log_to_file:
                self.logger.info(f"Analyzed text using OpenAI: {result}")
            return result
        except (OpenAIError, RateLimitError, BadRequestError) as e:
            self.logger.error(f"OpenAI error: {e}. Falling back to Hugging Face...")
            return self.analyze_text_huggingface(text)

    def analyze_text_huggingface(self, text):
        """
        Analyzes text using Hugging Face's sentiment analysis as a fallback.
        :param text: The text to analyze.
        :return: Sentiment analysis result.
        """
        try:
            result = self.hugging_face_pipeline(text)
            self.logger.info(f"Analyzed text using Hugging Face: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Hugging Face error: {e}. Falling back to scikit-learn...")
            return self.analyze_text_scikit(text)

    def analyze_text_scikit(self, text):
        """
        Analyzes text using a simple Scikit-learn classifier (fallback).
        :param text: The text to analyze.
        :return: Classification result.
        """
        try:
            text_vectorized = self.vectorizer.transform([text])
            prediction = self.classifier.predict(text_vectorized)
            result = f"Scikit-learn classification: {prediction[0]}"
            self.logger.info(f"Analyzed text using Scikit-learn: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Scikit-learn error: {e}. No further fallbacks available.")
            raise

    def train_scikit_model(self, X_train, y_train):
        """
        Trains the Scikit-learn Naive Bayes classifier on provided training data.
        :param X_train: The training text data.
        :param y_train: The training labels.
        """
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)
        self.logger.info("Trained Scikit-learn classifier.")

