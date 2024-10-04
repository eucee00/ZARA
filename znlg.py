# hello_world/ZB/znlg.py
# This is Zira's Natural Language Generation (NLG) handler, supporting dynamic API fallbacks.

import json

import openai
from openai import BadRequestError, OpenAIError, RateLimitError
from transformers import pipeline

from utils.logger import Logger


class ZNlg:
    def __init__(self, config_path="hello_world/config.json", log_file_path='logs/znlg.log'):
        """
        Initializes the ZNlg instance with API keys and logger for response generation.
        
        :param config_path: Path to the configuration file containing the API keys.
        :param log_file_path: Path to the log file for logging NLG operations.
        """
        self.config = self.load_config(config_path)
        self.logger = Logger(log_file_path=log_file_path)
        
        # API keys for multiple services
        self.openai_api_key = self.config.get("openai_api_key")
        self.huggingface_api_url = self.config.get("huggingface_api_url")

        # Initialize OpenAI API if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("OpenAI API initialized.")
        else:
            self.logger.error("OpenAI API key is missing.")
        
        # Initialize Hugging Face model if available
        if self.huggingface_api_url:
            self.hugging_face_generator = pipeline("automatic-speech-recognition", model="openai/whisper-large")
            self.logger.info("Hugging Face text-generation pipeline initialized.")
        
        self.logger.info("Zira NLG initialized with available APIs.")

    def load_config(self, path):
        """
        Loads the configuration file and returns the API keys and other settings.
        
        :param path: Path to the configuration file containing the API keys.
        :return: Parsed configuration as a dictionary.
        :raises: FileNotFoundError, JSONDecodeError, ValueError for any loading errors.
        """
        try:
            with open(path, "r") as file:
                config = json.load(file)
                return config
        except FileNotFoundError:
            self.logger.error("Configuration file not found.")
            raise
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def generate_response_openai(self, prompt, engine="davinci", max_tokens=200, temperature=0.7, n=1, stop=None):
        """
        Generates a response using OpenAI's Completion API. Falls back to Hugging Face if OpenAI is unavailable.
        
        :param prompt: The prompt text for the language generation model.
        :param engine: The OpenAI model engine to use for generation (default: 'davinci').
        :param max_tokens: Maximum number of tokens to generate (default: 200).
        :param temperature: Controls response randomness (default: 0.7).
        :param n: Number of responses to generate (default: 1).
        :param stop: Sequence at which to stop response generation (default: None).
        :return: The generated text response.
        """
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                stop=stop
            )
            generated_text = response.choices[0].text.strip()
            self.logger.info(f"Generated response using OpenAI for prompt: {prompt}")
            return generated_text
        except (OpenAIError, RateLimitError, BadRequestError) as e:
            self.logger.error(f"OpenAI error: {e}. Falling back to Hugging Face...")
            return self.generate_response_huggingface(prompt)

    def generate_response_huggingface(self, prompt, max_length=200, temperature=0.7):
        """
        Generates a response using Hugging Face's text generation model as a fallback.
        
        :param prompt: The prompt text for the language generation model.
        :param max_length: Maximum length of the generated text (default: 200).
        :param temperature: Controls response randomness (default: 0.7).
        :return: The generated text response.
        """
        try:
            response = self.huggingface_generator(prompt, max_length=max_length, temperature=temperature)
            generated_text = response[0]['generated_text'].strip()
            self.logger.info(f"Generated response using Hugging Face for prompt: {prompt}")
            return generated_text
        except Exception as e:
            self.logger.error(f"Hugging Face error: {e}. No further fallbacks available.")
            return "Error generating response with fallback."

    def generate_response(self, prompt, engine="davinci", max_tokens=200, temperature=0.7, n=1, stop=None):
        """
        Wrapper method for generating a response using OpenAI and falling back to Hugging Face if needed.
        
        :param prompt: The prompt text for the language generation model.
        :param engine: The OpenAI model engine to use for generation (default: 'davinci').
        :param max_tokens: Maximum number of tokens to generate (default: 200).
        :param temperature: Controls response randomness (default: 0.7).
        :param n: Number of responses to generate (default: 1).
        :param stop: Sequence at which to stop response generation (default: None).
        :return: The generated text response.
        """
        # Try OpenAI first, then fall back to Hugging Face
        return self.generate_response_openai(prompt, engine=engine, max_tokens=max_tokens, temperature=temperature, n=n, stop=stop)

    def generate_conversational_response(self, messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=200):
        """
        Generates a response for a conversation using OpenAI's Chat API. If unavailable, falls back to Hugging Face.
        
        :param messages: A list of conversation messages for context.
        :param model: The OpenAI model to use for chat (default: "gpt-3.5-turbo").
        :param temperature: Controls response randomness (default: 0.7).
        :param max_tokens: Maximum number of tokens in the response (default: 200).
        :return: The conversational response.
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            generated_text = response['choices'][0]['message']['content'].strip()
            self.logger.info(f"Generated conversational response using OpenAI.")
            return generated_text
        except (OpenAIError, RateLimitError, BadRequestError) as e:
            self.logger.error(f"OpenAI Chat API error: {e}. No fallback for chat models.")
            return "Error generating conversational response."
