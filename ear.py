# sensors/ear.py
# This is Zara's medium of hearing, converting speech to text for better processing
#import logging as Logger
import speech_recognition as sr

from utils.logger import Logger

class Ear:
    def __init__(self, logger=None):
        """
        Initializes the Ear object.
        :param logger: Logger instance to log actions and errors.
        """
        
        '''
        #initialize logger
        self.logger = Logger
        '''
        
        # Ensure logger is always initialized
        
        if logger is None:
            logger = Logger(log_file_path='logs/ear.log')
        
        self.logger = logger  # Ensure logger is assigned

        # Debugging to verify logger is correctly set
        self.logger.info("Ear sensor initialized with logger.")
        


        self.recognizer = sr.Recognizer()
        self.microphone = self._initialize_microphone()

    def _initialize_microphone(self):
        """
        Initializes the microphone object and handles errors in case the microphone is unavailable.
        :return: A Microphone object or None if initialization fails.
        """
        try:
            microphone = sr.Microphone()
            self.logger.info("Microphone initialized successfully.")
            return microphone
        except Exception as e:
            self.logger.error(f"Error initializing microphone: {str(e)}")
            return None

    def listen_once(self, timeout=None):
        """
        Listens for a single phrase and converts it to text.
        :param timeout: Maximum time to listen for speech (None means indefinite).
        :return: The recognized speech as text, or None if an error occurs.
        """
        if not self.microphone:
            self.logger.error("Microphone is not available.")
            return None

        with self.microphone as source:
            try:
                self.logger.info("Listening for a single phrase...")
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognize_speech(audio)
                return text
            except sr.WaitTimeoutError:
                self.logger.warning("Listening timed out.")
                #if timed out, listen again
                return self.listen_once(timeout=timeout)  # Retry listening if timed out
            except sr.NoMicrophoneError:
                self.logger.error("No microphone detected.")
                return None
            except Exception as e:
                self.logger.error(f"An error occurred while listening: {str(e)}")
                return None
        
        

    def recognize_speech(self, audio):
        """
        Converts audio to text using Google's speech recognition API.
        :param audio: The recorded audio.
        :return: Recognized text or None if an error occurs.
        """
        try:
            self.logger.info("Setting up Google Speech Recognition...")
            self.logger.info("Recognizing speech...")
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Recognized speech: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Speech recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None
'''
#test
ear = Ear()
text = ear.listen_once()
print(text)
'''