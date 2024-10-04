# main.py
# The main entry point for starting and managing Zara's lifecycle.

from hello_world.zara import Zara
from utils.logger import Logger


def main():
    """
    The main function that starts and manages Zara's lifecycle.
    """
    logger = Logger(log_file_path='logs/main.log')
    zara = None
    
    # Log the start of Zara
    logger.info("Starting Zara...")
    

    try:
        # Initialize Zara
        logger.info("Initializing Zara...")
        zara = Zara()

        # Run Zara's main interaction loop
        logger.info("Zara is now ready and waiting for the wake word.")
        zara.run()

    except KeyboardInterrupt:
        # Handle graceful shutdown when Ctrl+C is pressed
        logger.info("Shutting down Zara due to keyboard interruption.")
    except Exception as e:
        # Log any errors that occur during the process
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        # Ensure Zara is properly reset before exiting
        if zara:
            logger.info("Resetting Zara before exiting.")
            zara.reset()


if __name__ == "__main__":
    main()
