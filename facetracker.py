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
            if self.compare_face_data(stored_face_data, captured_face_data):
                self.show_message(f"Welcome {stored_usernames[0]}!")  # Assuming only one user for simplicity
            else:
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
                return face_data

    def save_user_data(self, username, face_data):
        if face_data is not None:
            data_dir = 'userdata'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            xml_file = os.path.join(data_dir, f'{username}.xml')

            root = ET.Element("User")
            ET.SubElement(root, "Username").text = username
            face_data_str = ','.join(str(pixel) for row in face_data for pixel in row)
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
                    face_data_list = [int(pixel) for pixel in face_data_str.split(',')]
                    face_data = np.array(face_data_list, dtype=np.uint8).reshape(-1, len(face_data_list) // len(face_data_list))
                    stored_usernames.append(username)
                    stored_face_data.append(face_data)
        return stored_usernames, stored_face_data

    def compare_face_data(self, stored_face_data, captured_face_data):
        # Convert captured face data to grayscale if necessary
        if len(captured_face_data.shape) > 2:
            captured_face_data = cv2.cvtColor(captured_face_data, cv2.COLOR_BGR2GRAY)

        # Initialize LBPH Face Recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Train recognizer with all stored face data
        labels = np.arange(len(stored_face_data))
        face_recognizer.train(stored_face_data, labels)

        # Predict the label of the captured face data
        label, _ = face_recognizer.predict(captured_face_data)

        # If the predicted label matches any stored label, the faces are considered the same
        return label in labels
        
    def show_message(self, message):
        self.message_label.text = message

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    FaceDetectionApp().run()
