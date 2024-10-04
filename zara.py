# hello_world/zara.py
# Zara's main script for wake-up detection and interaction

import time
from sensors.ear import Ear
from sensors.mouth import Mouth
from utils.logger import Logger
from .hz.hello import HelloZara
from .hz.wakezara import Wakezara
from random import choice


class Zara:
    def __init__(self):
        """
        Initializes the Zara class with wake-up detection and conversational capabilities.
        """
        self.logger = Logger(log_file_path='logs/zara.log')
        self.wakezara = Wakezara()
        self.hello_zara = HelloZara()
        self.mouth = Mouth()
        self.logger.info("Zara initialized and ready to wake up.")

    def run(self):
        """
        Orchestrates the process of listening for a wake word, and upon detection,
        hands over control to HelloZara for interaction.
        """
        self.logger.info("Zara is waiting for the wake word...")
        print("Zara is waiting on Wakeword")

        # Step 1: Listen for the wake word
        while not self.wakezara.is_awake():
            wakey = self.wakezara.detect_wakezara()
            if wakey:
                self.logger.info("User woke up Zara.")
                print("User woke up Zara")
                break

        # Step 2: Greet once awake
        self.logger.info("Wake word detected. Activating Zara...")
        self.hello_zara.greet()

        # Step 3: Start interaction loop, Zara waits for user input
        timeout_duration = 30  # Wait for 30 seconds for the user's input
        idle_time = 0
        while self.wakezara.is_awake():
            try:
                user_input = self.hello_zara.ear.listen_once(timeout=5)
                
                if user_input:
                    # If user speaks, process the input and respond
                    response = self.hello_zara.listen_and_respond(user_input)
                    self.logger.info(f"Response generated: {response}")
                    idle_time = 0  # Reset idle time if user speaks
                else:
                    idle_time += 5  # Increment idle time by 5 seconds for each listen cycle
                    self.logger.info(f"Idle for {idle_time} seconds...")

                    if idle_time >= timeout_duration:
                        # Zara waits 30 seconds for input and then asks the user why she was woken up
                        random_messages = [
                            "Why did you wake me up if you have nothing to say?",
                            "You called me, and now you're quiet. What's going on?",
                            "Are you just playing with me? What do you need?"
                        ]
                        random_angry_message = [
                            "If you woke me up for nothing, I'm going back to sleep now.",
                            "Next time, don't wake me up for no reason. I'm going back to sleep."
                        ]
                        
                        # Use NLG to ask why the user woke Zara up
                        random_prompt = choice(random_messages)
                        angry_prompt = choice(random_angry_message)

                        # Zara speaks her frustration
                        self.logger.info(f"No input detected for {timeout_duration} seconds. Asking user.")
                        self.mouth.speak(random_prompt)
                        
                        # Wait another 10 seconds
                        further_input = self.hello_zara.ear.listen_once(timeout=10)
                        if not further_input:
                            # After an additional 10 seconds, Zara gets "angry" and goes back to sleep
                            self.mouth.speak(angry_prompt)
                            self.logger.info("No further input detected. Zara is going back to sleep.")
                            self.wakezara.reset_wake_state()  # Zara goes back to sleep
                            break
                        else:
                            # If user responds after Zara's prompt, process their input
                            response = self.hello_zara.listen_and_respond(further_input)
                            self.logger.info(f"Response generated: {response}")
                            idle_time = 0  # Reset idle time after response
            except Exception as e:
                self.logger.error(f"An error occurred during interaction: {str(e)}")
                self.wakezara.reset_wake_state()  # Reset if an error occurs

        self.logger.info("Zara has stopped listening and returned to the idle state.")

    def reset(self):
        """
        Resets Zara's state, allowing her to be woken up again.
        """
        self.wakezara.reset_wake_state()
        self.logger.info("Zara has been reset and is ready to wake up again.")
