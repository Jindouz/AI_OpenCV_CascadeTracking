import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock


class MotionDetectorApp(App):
    def build(self):
        self.title = 'Motion Detector'
        self.video_source = 0  # Use default camera
        self.detected_label = Label(text="Not Detected", color=(1, 1, 1, 1))
        self.sensitivity_slider = Slider(min=1, max=100, value=50)
        self.sensitivity_slider.bind(value=self.on_sensitivity_change)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.detected_label)
        layout.add_widget(self.sensitivity_slider)

        self.capture = cv2.VideoCapture(self.video_source)
        ret, frame = self.capture.read()

        if ret:
            self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Change this to your cascade classifier

            self.video_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            self.video_texture.flip_vertical()

            self.image = Image(texture=self.video_texture)
            layout.add_widget(self.image)

            Clock.schedule_interval(self.detect_motion, 1.0 / 30.0)  # Call detect_motion every 1/30th of a second (30 fps)

        return layout

    def detect_motion(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_size = int(self.sensitivity_slider.value)  # Convert slider value to integer
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))

            if len(faces) > 0:
                self.detected_label.text = "Detected"
                self.detected_label.color = (1, 0, 0, 1)  # Red background for detected
            else:
                self.detected_label.text = "Not Detected"
                self.detected_label.color = (1, 1, 1, 1)  # White background for not detected

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around detected object

            buf = frame.tobytes()
            self.video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def on_sensitivity_change(self, instance, value):
        # No need to implement now, because sensitivity is already adjusted in detect_motion
        pass



if __name__ == '__main__':
    MotionDetectorApp().run()
