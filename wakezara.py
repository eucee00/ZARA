# hello_world/hz/wakezara.py
# This script detects the wake word ("Hey Zara") using the Ear class.

from sensors.ear import Ear
from utils.logger import Logger

class Wakezara:
    def __init__(self, wake_word=("Hey", "Zara"), timeout=10):
        """
        Initialize the Wakezara class with an Ear instance and a Logger.
        
        :param wake_word: The phrase that triggers Zara to wake up (default: ("Hey", "Zara")).
        :param timeout: The maximum time to wait for a wake word (default: 10 seconds).
        """
        self.ear = Ear()
        self.logger = Logger(log_file_path='logs/wakezara.log')
        self.wake_word = tuple(word.lower() for word in wake_word)  # Convert all words to lowercase
        self.wakezara_detected = False
        self.timeout = timeout
        self.logger.info(f"Wakezara initialized with wake word: {self.wake_word}")

    def detect_wakezara(self):
        """
        Listens for a specific wake word using the Ear class. If detected, sets wakezara_detected to True.
        """
        self.logger.info("Listening for wake word...")
        
        recognized_text = self.ear.listen_once(timeout=self.timeout)
        
        if recognized_text:
            self.logger.info(f"Recognized text: {recognized_text}")
            
            # Check if all words in wake_word are present in the recognized text
            if all(word in recognized_text.lower() for word in self.wake_word):
                self.wakezara_detected = True
                self.logger.info("Wake word detected! Zara is now active.")
            else:
                self.logger.info("Wake word not detected.")
        else:
            self.logger.info("No recognizable speech detected.")

    def is_awake(self):
        """
        Returns the current wake state of Zara.
        
        :return: Boolean indicating whether the wake word has been detected.
        """
        return self.wakezara_detected
