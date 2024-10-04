# sensors/eye.py
# This is Zara's eye to capture visuals using a camera

import cv2
import time
import threading
from utils.logger import Logger

class Eye:
    def __init__(self, camera_index=0, logger=None):
        """
        Initializes the Eye (camera) object.
        :param camera_index: The index of the camera (default is 0).
        :param logger: Logger instance for logging actions and errors.
        """
        if logger is None:
            logger = Logger(log_file_path='logs/eye.log')
        self.logger = logger

        if hasattr(self, 'logger'):
            self.logger.info('ESI Success.')
        else:
            print("ESI failed.")
        self.camera_index = camera_index
        self.capture = None
        self.is_streaming = False
        self.video_writer = None

    def open_camera(self):
        """
        Opens the camera for capturing video.
        :return: True if the camera is opened successfully, False otherwise.
        """
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.logger.error(f"Unable to open camera at index {self.camera_index}")
            return False
        
        self.logger.info(f"Camera {self.camera_index} opened successfully.")
        return True

    def capture_frame(self):
        """
        Captures a single frame from the camera.
        :return: Captured frame (image) or None if there's an error.
        """
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.logger.info("Captured a frame successfully.")
                return frame
            else:
                self.logger.error("Failed to capture a frame.")
                return None
        else:
            self.logger.error("Camera is not open for capturing.")
            return None

    def save_image(self, frame, file_path='images/captured_image.jpg'):
        """
        Saves the given frame (image) to a file.
        :param frame: The image/frame to save.
        :param file_path: The path where the image will be saved.
        """
        try:
            if frame is not None:
                cv2.imwrite(file_path, frame)
                self.logger.info(f"Image saved at {file_path}")
            else:
                self.logger.warning("Cannot save an empty frame.")
        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")

    def start_video_stream(self, file_path='videos/captured_video.avi', fps=20.0, frame_size=(640, 480), codec='XVID'):
        """
        Starts capturing and saving video to a file.
        :param file_path: Path to save the video file.
        :param fps: Frames per second.
        :param frame_size: Size of the video frames.
        :param codec: Video codec to use (default is XVID).
        """
        if not self.open_camera():
            self.logger.error("Cannot start video stream; camera failed to open.")
            return

        # VideoWriter to save the video
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.video_writer = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

        self.is_streaming = True
        threading.Thread(target=self._stream_video).start()

    def _stream_video(self):
        """
        Private method to stream video in the background and save it.
        """
        self.logger.info("Starting video stream...")
        while self.is_streaming and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.video_writer.write(frame)
                # Optionally show the frame in a window
                cv2.imshow('Zara Eye', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Video stream stopped by user.")
                    break
            else:
                self.logger.error("Error while reading frames during video streaming.")
                break

        self.stop_video_stream()

    def stop_video_stream(self):
        """
        Stops the video streaming and releases resources.
        """
        if self.is_streaming:
            self.is_streaming = False
            self.logger.info("Stopping video stream...")

            if self.capture:
                self.capture.release()
            if self.video_writer:
                self.video_writer.release()

            cv2.destroyAllWindows()
            self.logger.info("Video stream and resources released.")

    def release(self):
        """
        Releases the camera and any other resources.
        """
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        self.logger.info("Camera resources released.")
