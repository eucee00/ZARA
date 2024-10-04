# sensors/mouth.py
# This controls Zara's voice, speaking, and vocal outputs

import os
import threading
import pyttsx3
from utils.logger import Logger


class Mouth:
    def __init__(self, rate=200, volume=1.0, logger=None):
        """
        Initializes the Mouth object for text-to-speech functionality.
        :param rate: Speed of speech (default is 200 words per minute).
        :param volume: Volume level (default is 1.0, max volume).
        :param logger: Logger instance for logging actions and errors.
        """
        if logger is None:
            logger = Logger(log_file_path='logs/mouth.log')
        self.logger = logger

        if hasattr(self, 'logger'):
            self.logger.info('MSI Success.')
        else:
            print("MSI failed.")
        
        # Initialize the pyttsx3 engine and handle exceptions if initialization fails
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            self.logger.error(f"Error initializing text-to-speech engine: {str(e)}")
            raise e
        
        # Set speech rate and volume with validation
        self.set_rate(rate)
        self.set_volume(volume)

        # Set the default voice to Zira (voice_id=1) or custom based on system capabilities
        self.set_voice(1)
        
        self.logger.info("Mouth initialized with rate: {}, volume: {}".format(rate, volume))

    def set_rate(self, rate):
        """
        Sets the speech rate (words per minute).
        :param rate: The desired speech rate.
        """
        try:
            self.engine.setProperty('rate', rate)
            self.logger.info(f"Speech rate set to {rate} WPM.")
        except Exception as e:
            self.logger.error(f"Error setting speech rate: {str(e)}")

    def set_volume(self, volume):
        """
        Sets the speech volume.
        :param volume: The desired volume level (0.0 to 1.0).
        """
        if not (0.0 <= volume <= 1.0):
            self.logger.error("Volume must be between 0.0 and 1.0.")
            return
        
        try:
            self.engine.setProperty('volume', volume)
            self.logger.info(f"Volume set to {volume * 100}%.")
        except Exception as e:
            self.logger.error(f"Error setting volume: {str(e)}")

    def set_voice(self, voice_id):
        """
        Sets the voice for the speech engine based on the voice_id.
        :param voice_id: The index of the voice (1 is typically Microsoft Zira).
        """
        voices = self.engine.getProperty('voices')
        if 0 <= voice_id < len(voices):
            try:
                voice = voices[voice_id]
                self.engine.setProperty('voice', voice.id)
                self.logger.info(f"Voice set to {voice.name}.")
            except Exception as e:
                self.logger.error(f"Error setting voice: {str(e)}")
        else:
            self.logger.warning(f"Voice ID {voice_id} not found. Using default voice.")

    def list_voices(self):
        """
        Lists all available voices on the system.
        :return: List of voice names and IDs.
        """
        voices = self.engine.getProperty('voices')
        voice_list = [{'name': voice.name, 'id': voice.id} for voice in voices]
        return voice_list

    def speak(self, text):
        """
        Speaks the provided text using text-to-speech.
        :param text: The text to be spoken.
        """
        self.logger.info(f"Speaking: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error during speech: {str(e)}")

    def speak_async(self, text):
        """
        Speaks the provided text asynchronously, allowing other operations to continue.
        :param text: The text to be spoken.
        """
        threading.Thread(target=self.speak, args=(text,)).start()

    def stop_speaking(self):
        """
        Stops any ongoing speech immediately.
        """
        try:
            self.engine.stop()
            self.logger.info("Speech stopped.")
        except Exception as e:
            self.logger.error(f"Error stopping speech: {str(e)}")

    def save_speech_to_file(self, text, file_path):
        """
        Converts text to speech and saves it as an audio file (e.g., .mp3 or .wav).
        :param text: The text to convert to speech.
        :param file_path: The path to save the audio file.
        """
        self.logger.info(f"Saving speech to {file_path}")
        try:
            self.engine.save_to_file(text, file_path)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error saving speech to file: {str(e)}")

    def __del__(self):
        """
        Destructor to ensure the engine is properly stopped.
        """
        try:
            self.engine.stop()
            self.logger.info("Mouth engine stopped safely.")
        except Exception as e:
            self.logger.error(f"Error while shutting down Mouth engine: {str(e)}")

