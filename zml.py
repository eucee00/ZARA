# hello_world/ZB/zml.py
# This is Zira's Machine Learning (ML) handler, supporting traditional ML models and dynamic API fallbacks.

import json
import openai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.logger import Logger
from openai import OpenAIError, RateLimitError, BadRequestError


class ZMl:
    def __init__(self, path="hello_world/config.json", log_file_path='logs/zml.log'):
        """
        Initializes the ZMl instance, sets up API key, logger, and scikit-learn models.
        
        :param path: Path to the configuration file containing the API keys.
        :param log_file_path: Path to the log file for logging ML operations.
        """
        self.config = self.load_config(path)
        self.logger = Logger(log_file_path=log_file_path)
        
        # OpenAI API setup
        self.openai_api_key = self.config.get("openai_api_key")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("OpenAI API initialized.")
        else:
            self.logger.error("OpenAI API key is missing.")
        
        # Initialize Scikit-learn models
        self.vectorizer = TfidfVectorizer()
        self.regression_model = LinearRegression()
        self.classifier = MultinomialNB()

    def load_config(self, path):
        """
        Loads the configuration file and returns the API keys and other settings.
        
        :param path: Path to the configuration file containing the API keys.
        :return: Parsed configuration as a dictionary.
        :raises: FileNotFoundError, JSONDecodeError, ValueError for any loading errors.
        """
        try:
            with open(path, 'r') as file:
                config = json.load(file)
                return config
        except FileNotFoundError:
            self.logger.error("Configuration file not found.")
            raise
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def perform_ml_task_openai(self, task_description, input_data, output_data, model_type="text-davinci-003",
                               max_tokens=100, temperature=0.7):
        """
        Executes a machine learning task using OpenAI's completion models.
        
        :param task_description: A brief description of the ML task.
        :param input_data: The input data for the task.
        :param output_data: The expected output or guidance for the task.
        :param model_type: The OpenAI model engine to use (default: 'text-davinci-003').
        :param max_tokens: Maximum tokens for the model response (default: 100).
        :param temperature: Sampling temperature to control randomness (default: 0.7).
        :return: The model's response as a string.
        """
        try:
            prompt = (
                f"Task: {task_description}/n/n"
                f"Input: {input_data}/n/n"
                f"Output: {output_data}/n/n"
                f"Generate a response using the provided input and output."
            )
            response = openai.Completion.create(
                engine=model_type,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            self.logger.info(f"Performed ML task with OpenAI: {task_description}")
            return generated_text
        except (OpenAIError, RateLimitError, BadRequestError) as e:
            self.logger.error(f"OpenAI error: {e}. Falling back to Scikit-learn...")
            return self.perform_ml_task_scikit(input_data, output_data)

    def perform_ml_task_scikit(self, X, y, task_type="classification"):
        """
        Executes a machine learning task using Scikit-learn models for classification or regression.
        
        :param X: Input data (features).
        :param y: Output data (labels).
        :param task_type: The type of ML task ('classification' or 'regression').
        :return: Performance metrics (accuracy for classification, R^2 for regression).
        """
        try:
            if task_type == "classification":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                X_train_vectorized = self.vectorizer.fit_transform(X_train)
                X_test_vectorized = self.vectorizer.transform(X_test)

                # Train and predict using Naive Bayes Classifier
                self.classifier.fit(X_train_vectorized, y_train)
                predictions = self.classifier.predict(X_test_vectorized)
                accuracy = accuracy_score(y_test, predictions)
                self.logger.info(f"Performed classification with Scikit-learn. Accuracy: {accuracy}")
                return f"Classification accuracy: {accuracy}"
            
            elif task_type == "regression":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                self.regression_model.fit(X_train, y_train)
                r_squared = self.regression_model.score(X_test, y_test)
                self.logger.info(f"Performed regression with Scikit-learn. R^2: {r_squared}")
                return f"Regression R^2: {r_squared}"
        
        except Exception as e:
            self.logger.error(f"Error performing Scikit-learn ML task: {e}")
            return "Error: Failed to perform ML task with Scikit-learn"

    def train_scikit_model(self, X, y, task_type="classification"):
        """
        Trains the Scikit-learn model for classification or regression tasks.
        
        :param X: Input data (features).
        :param y: Output data (labels).
        :param task_type: The type of ML task ('classification' or 'regression').
        """
        try:
            if task_type == "classification":
                X_vectorized = self.vectorizer.fit_transform(X)
                self.classifier.fit(X_vectorized, y)
                self.logger.info("Trained Scikit-learn Naive Bayes classifier.")
            elif task_type == "regression":
                self.regression_model.fit(X, y)
                self.logger.info("Trained Scikit-learn Linear Regression model.")
        except Exception as e:
            self.logger.error(f"Error training Scikit-learn model: {e}")
            raise

