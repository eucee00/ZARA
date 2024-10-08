# hello_world/hz/hello.py
# This is Zara's hello script, using the mouth to speak and ears to hear.

from hello_world.ZB.zml import ZMl
from hello_world.ZB.znlg import ZNlg
from hello_world.ZB.znlp import ZbNlp
from sensors.ear import Ear
from sensors.mouth import Mouth
from utils.logger import Logger


class HelloZara:
    def __init__(self):
        """
        Initialize Zara's Hello script, setting up the sensors and the ZB components.
        """

        self.logger = Logger(log_file_path='logs/hellozara.log')
        
        # Initialize sensors
        self.ear = Ear(
            logger=self.logger
        )
        self.mouth = Mouth(
            logger=self.logger
        )
        
        # Initialize ZB components for NLP, NLG, and ML
        self.nlp = ZbNlp()
        self.nlg = ZNlg()
        self.ml = ZMl()

        self.logger.info("HelloZara initialized.")

    def listen_and_respond(self, user_input=None):
        """
        Process user input, use ZbNlp for analysis, ZMl for any ML tasks, 
        and ZNlg to generate a response to be spoken by Zara.
        
        :param user_input: Input string from the user. If None, use Ear to capture input.
        :return: The response generated by Zara.
        """
        try:
            # If no user_input provided, Zara listens using the Ear sensor
            if not user_input:
                user_input = self.ear.listen_once(timeout=5)
                if not user_input:
                    self.logger.info("No input received from the user.")
                    return "Sorry, I didn't catch that."

            # Step 1: Zara hears the input
            self.logger.info(f"User said: {user_input}")
            processed_text = self.nlp.analyze_text(user_input)
            
            # Step 2: Perform any specific ML task if required by the context (for example, intent classification)
            # Here, we're passing a generic task for demonstration purposes
            ml_task_description = "Classify user intent based on input"
            classified_intent = self.ml.perform_machine_learning(ml_task_description, user_input, processed_text)

            # Step 3: Generate response using ZNlg based on intent or analysis
            response = self.nlg.generate_response(f"Based on input '{user_input}', respond with {classified_intent}.")
            self.logger.info(f"Zara's response: {response}")

            # Step 4: Zara speaks the response
            self.mouth.speak(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error in listening and responding: {str(e)}")
            return "I'm sorry, something went wrong."

    def greet(self):
        """
        Zara greets the user, using NLG to generate a greeting message.
        
        :return: The greeting message.
        """
        try:
            # Generate a greeting response
            greeting_prompt = "Generate a friendly greeting for Zara to say."
            greeting_message = self.nlg.generate_response(greeting_prompt)
            self.logger.info(f"Zara greets: {greeting_message}")
            
            # Zara speaks the greeting
            self.mouth.speak(greeting_message)
            
            return greeting_message
        except Exception as e:
            self.logger.error(f"Error in generating greeting: {str(e)}")
            return "Hello!"
