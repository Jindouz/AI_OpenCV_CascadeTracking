import base64
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class FaceDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Create text input for username
        self.username_input = TextInput(hint_text='Enter Username', multiline=False)
        self.layout.add_widget(self.username_input)

        # Create buttons for registration and login
        self.register_button = Button(text='Register', size_hint=(None, None))
        self.register_button.bind(on_press=self.register_user)
        self.layout.add_widget(self.register_button)

        self.login_button = Button(text='Login', size_hint=(None, None))
        self.login_button.bind(on_press=self.login_user)
        self.layout.add_widget(self.login_button)

        # Create image widget for displaying video feed
        self.image = Image()
        self.layout.add_widget(self.image)

        # Create label for displaying messages
        self.message_label = Label(text='', size_hint=(1, None))
        self.layout.add_widget(self.message_label)

        # Start video capture and schedule update method
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def register_user(self, instance):
        username = self.username_input.text.strip()
        if username:
            face_data = self.capture_face_data()
            self.save_user_data(username, face_data)
            self.show_message(f"User '{username}' registered.")
        else:
            self.show_message("Please enter a valid username.")

    def login_user(self, instance):
        stored_usernames, stored_face_data = self.load_all_user_data()
        captured_face_data = self.capture_face_data()
        if captured_face_data is not None:
            for username, stored_data in zip(stored_usernames, stored_face_data):
                if self.compare_face_data(stored_data, captured_face_data):
                    self.show_message(f"Welcome {username}!")
                    return
            self.show_message("Face verification failed. Please try again.")
        else:
            self.show_message("Failed to capture face data. Please try again.")

    def capture_face_data(self):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_data = gray[y:y+h, x:x+w]

                # Ensure face_data is contiguous
                face_data = np.ascontiguousarray(face_data)

                return face_data

    def save_user_data(self, username, face_data):
        if face_data is not None:
            data_dir = 'userdata'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            xml_file = os.path.join(data_dir, f'{username}.xml')

            root = ET.Element("User")
            ET.SubElement(root, "Username").text = username
            face_data_str = base64.b64encode(face_data.tobytes()).decode('utf-8')
            ET.SubElement(root, "FaceData").text = face_data_str

            tree = ET.ElementTree(root)
            tree.write(xml_file)

    def load_all_user_data(self):
        data_dir = 'userdata'
        stored_usernames = []
        stored_face_data = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(".xml"):
                    xml_file = os.path.join(data_dir, filename)
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    username = root.find("Username").text
                    face_data_str = root.find("FaceData").text

                    # Define height and width based on the shape of the face data
                    height = 100  # Define the appropriate height here
                    width = 100   # Define the appropriate width here

                    # Decode face data from base64 and reshape
                    face_data = np.frombuffer(base64.b64decode(face_data_str), dtype=np.uint8).reshape(height, width)
                    stored_usernames.append(username)
                    stored_face_data.append(face_data)
        return stored_usernames, stored_face_data

    def compare_face_data(self, face_data1, face_data2):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train([face_data1], np.array([1]))
        label, confidence = face_recognizer.predict(face_data2)
        # Assuming a label of 1 means a match and confidence level less than some threshold
        threshold = 70  # Set your desired threshold value here
        return label == 1 and confidence < threshold
    
    def show_message(self, message):
        self.message_label.text = message

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    FaceDetectionApp().run()
