import time
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
import threading
import queue
import pygame

class MotionDetectorApp(App):
    def build(self):
        self.title = 'Motion Detector'
        self.frame_queue = queue.Queue()
        self.gray = None
        self.detected_label = Label(text="Not Detected", color=(1, 1, 1, 1))
        self.sensitivity_slider = Slider(min=1, max=100, value=50)
        self.sensitivity_slider.bind(value=self.on_sensitivity_change)
        self.last_sound_time = 0  


        self.layout = BoxLayout(orientation='vertical')
        self.layout.add_widget(self.detected_label)
        self.layout.add_widget(self.sensitivity_slider)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
        self.capture.set(cv2.CAP_PROP_FPS, 15)  # Reduced frame rate

        ret, frame = self.capture.read()
        if ret:
            # self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # self.detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')
            self.detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

            self.video_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            self.video_texture.flip_vertical()

            self.image = Image(texture=self.video_texture)
            self.layout.add_widget(self.image)

            Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update display at 30 fps
            threading.Thread(target=self.capture_frames).start()  # Start the frame capture thread

        self.stop_event = threading.Event()  # Event to signal thread to stop
        self.thread = threading.Thread(target=self.capture_frames)
        self.thread.daemon = True  # Set thread as daemon
        self.thread.start()
        self.display_buffer = queue.Queue(maxsize=2)  # Buffer for display
        return self.layout

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                break
            if self.display_buffer.full():  # If the display buffer is full, skip
                continue
            self.frame_queue.put(frame)

    def update(self, dt):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.detect_motion(frame)
            self.display_buffer.put(frame)  # Put frame in display buffer

        if not self.display_buffer.empty():
            display_frame = self.display_buffer.get()
            buf = display_frame.tobytes()
            self.video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            del buf  # Manage the buffer

    def detect_motion(self, frame):
        min_size = int(self.sensitivity_slider.value)
        # Increase minNeighbors to reduce false positives
        faces = self.detector.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))

        if len(faces) > 0:
            self.detected_label.text = "Detected"
            self.detected_label.color = (1, 0, 0, 1)
            # Check if sound has already been played within the last 3 seconds
            if time.time() - self.last_sound_time >= 3:
                pygame.mixer.init()
                pygame.mixer.music.load("detected.mp3")
                pygame.mixer.music.set_volume(0.5)  # Set volume to 50%
                pygame.mixer.music.play()
                self.last_sound_time = time.time()
        else:
            self.detected_label.text = "Not Detected"
            self.detected_label.color = (1, 1, 1, 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        buf = frame.tobytes()
        self.video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def on_stop(self):
        self.stop_event.set()  # Signal the thread to stop
        self.thread.join()  # Wait for the thread to finish
        self.capture.release()  # Release the video capture

    def on_sensitivity_change(self, instance, value):
        pass  # Sensitivity adjustment logic can be added here if needed

if __name__ == '__main__':
    MotionDetectorApp().run()
