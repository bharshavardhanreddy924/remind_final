from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.storage.jsonstore import JsonStore
from kivy.uix.popup import Popup
import nltk
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty
from kivy.graphics import Color, RoundedRectangle
from random import sample
import speech_recognition as sr
from datetime import datetime
from threading import Thread
import time
from plyer import notification
import requests
from functools import partial
from kivy.properties import ObjectProperty
from datetime import datetime
import pyttsx3
import nltk
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import random
import numpy as np
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from kivy.uix.textinput import TextInput
import datetime
from kivy.core.audio import SoundLoader
import json
import os
ALARM_SOUND_PATH = 'C:/Users/borut/Desktop/hackathon/app/alarm.mp3'  # Default alarm sound path


from plyer import gps

recognizer = sr.Recognizer()
engine = pyttsx3.init()

engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except sr.WaitTimeoutError:
            return "timeout"
        except sr.UnknownValueError:
            return "unclear"
        except sr.RequestError:
            return "error"
        

class TaskItem(BoxLayout):
    task_text = StringProperty('')
    completed = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = '50dp'
        self.spacing = '10dp'
        self.padding = '5dp'

class PatternMatchGame(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 4  # 4x4 grid
        self.padding = 20
        self.spacing = 5
        
        # Game state
        self.pattern = []
        self.user_pattern = []
        self.grid_size = 4
        self.buttons = []
        
        # Create grid buttons
        for i in range(self.grid_size ** 2):
            btn = Button(background_color=[1, 1, 1, 1])
            btn.index = i
            btn.bind(on_press=self.handle_button)
            self.buttons.append(btn)
            self.add_widget(btn)
        
        # Add message label
        self.message_label = Button(
            text="Watch the pattern!",
            font_size=24,
            size_hint=(1, 0.2),
            background_color=[0.8, 0.8, 0.8, 1],
            disabled=True
        )
        self.add_widget(self.message_label)
        
        Clock.schedule_once(lambda dt: self.start_new_round(), 1)

    def start_new_round(self):
        self.user_pattern = []
        pattern_length = len(self.pattern) + 1
        self.pattern = sample(range(self.grid_size ** 2), pattern_length)
        self.message_label.text = f"Round {pattern_length}: Watch the pattern!"
        self.show_pattern()

    def show_pattern(self, index=0):
        if index < len(self.pattern):
            btn = self.buttons[self.pattern[index]]
            btn.background_color = [0, 1, 0, 1]
            Clock.schedule_once(lambda dt: self.restore_button_color(btn), 0.5)
            Clock.schedule_once(lambda dt: self.show_pattern(index + 1), 0.6)
        else:
            self.message_label.text = "Your turn! Repeat the pattern."

    def restore_button_color(self, button):
        button.background_color = [1, 1, 1, 1]

    def handle_button(self, instance):
        if self.message_label.text != "Your turn! Repeat the pattern.":
            return

        self.user_pattern.append(instance.index)
        
        if self.user_pattern == self.pattern[:len(self.user_pattern)]:
            if len(self.user_pattern) == len(self.pattern):
                self.message_label.text = "Correct! Get ready for the next round."
                Clock.schedule_once(lambda dt: self.start_new_round(), 1)
        else:
            self.message_label.text = "Wrong! Game over."
            self.pattern = []
            Clock.schedule_once(lambda dt: self.start_new_round(), 2)

    def reset_game(self):
        self.pattern = []
        self.start_new_round()

class PersonalInfoQA:
    def __init__(self):
        # Modified TfidfVectorizer parameters for better matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.9
        )
        self.knowledge_base = []
        self.tfidf_matrix = None
        
        self.conversation_starters = [
            "I'd be happy to tell you about that.",
            "Let me share what I know about that.",
            "Here's what I can tell you.",
            "That's an interesting question.",
        ]
        
        self.unclear_responses = [
            "I didn't catch that. Could you please repeat?",
            "Sorry, I didn't understand. Could you say that again?",
            "I'm having trouble understanding. One more time?",
        ]
    
    def train(self, text):
        sentences = sent_tokenize(text)
        # Clean and preprocess the sentences
        cleaned_sentences = [sent.strip() for sent in sentences if sent.strip()]
        self.knowledge_base = cleaned_sentences
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_sentences)
    
    def generate_response(self, prompt):
        if not self.knowledge_base:
            return "I haven't been trained with any memories yet."
        
        if prompt in ["unclear", "timeout", "error"]:
            return random.choice(self.unclear_responses)
        
        try:
            # Clean the prompt
            cleaned_prompt = prompt.strip().lower()
            prompt_vector = self.vectorizer.transform([cleaned_prompt])
            
            # Calculate similarity with all sentences
            similarities = cosine_similarity(prompt_vector, self.tfidf_matrix).flatten()
            
            # Sort sentences by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Lower similarity threshold and get more sentences
            relevant_sentences = [self.knowledge_base[idx] for idx in sorted_indices[:3] if similarities[idx] > 0.05]
            
            if relevant_sentences:
                starter = random.choice(self.conversation_starters)
                response = " ".join(relevant_sentences)
                return f"{starter} {response}"
            else:
                # If no matches, try to give a generic response based on keywords
                keywords = {
                    'hobbies': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'hobby': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'intrests': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'intrest': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'work': 'I work as a software engineer at Tech Corp.',
                    'live': 'I live in San Francisco.',
                    'age': 'I am 28 years old.',
                    'childhood': 'I often reminisce about my childhood near the lake.',
                    'education': 'I graduated from MIT in 2018 with a degree in Computer Science.'
                }
                
                for key, response in keywords.items():
                    if key in cleaned_prompt:
                        return response
                        
                return "I don't have specific information about that. Feel free to ask me about my hobbies, work, education, or interests!"
                
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return "I'm not sure how to respond to that. Could you try asking in a different way?"

def run_voice_assistant():
    qa_system = PersonalInfoQA()
    
    training_text = """
    I am 28 years old.
    I am a software engineer at Tech Corp.
    My favourite color is blue.
    I live in San Francisco.
    I graduated from MIT in 2018 with a degree in Computer Science.
    I love hiking in the mountains during the weekends.
    I enjoy playing sports, especially basketball.
    I often reminisce about my childhood near the lake.
    I love photography, especially landscapes and cityscapes.
    I enjoy reading science fiction books, especially by Isaac Asimov and Philip K. Dick.
    My favourite movie is Inception, and I love watching mind-bending films.
    I once took a solo trip to Japan in 2020, which was one of the best experiences of my life.
    I like playing video games, especially strategy games.
    I once ran a marathon in 2021 and finished with a time of 4 hours and 15 minutes.
    I am fluent in both English and Spanish.
    I enjoy spending time at the beach and love watching the sunset.
    I love cooking, especially trying out new recipes.
    I enjoy giving back to the community and volunteer at a local animal shelter.
    My first car was a used Honda Civic, and I remember how excited I was to get it.
    I enjoy puzzles and riddles, and I often challenge myself with brain teasers.
    """
    qa_system.train(training_text)
    
    speak("Hello! I'm your personal assistant. You can ask me questions or have a conversation with me. Say 'goodbye' when you want to end our chat.")
    
    while True:
        user_input = listen()
        
        if user_input == "timeout":
            speak("I'm still listening. Feel free to ask me anything.")
            continue
            
        if user_input in ["goodbye", "bye", "exit", "quit"]:
            speak("Goodbye! It was nice talking to you.")
            break
            
        response = qa_system.generate_response(user_input)
        speak(response)
        time.sleep(0.5)


KV = '''
#:import Factory kivy.factory.Factory

<TaskItem>:
    canvas.before:
        Color:
            rgba: (0.95, 0.95, 0.95, 1) if not self.completed else (0.9, 0.9, 0.9, 0.5)
        Rectangle:
            pos: self.pos
            size: self.size
    
    CheckBox:
        size_hint_x: 0.1
        active: root.completed
        on_active: root.completed = self.active
    
    Label:
        text: root.task_text
        color: (0, 0, 0, 1) if not root.completed else (0.5, 0.5, 0.5, 1)
        text_size: self.size
        halign: 'left'
        valign: 'center'
        strikethrough: root.completed

<MainScreen>:
    name: 'main'
    location_label: location_label_id
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: 0.95, 0.95, 0.95, 1
            Rectangle:
                pos: self.pos
                size: self.size

        # Top bar with location and emergency button
        BoxLayout:
            size_hint_y: 0.06
            padding: 3
            spacing: 8
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15, 15, 15, 15]

            Label:
                id: location_label_id
                text: "📍 Fetching location..."
                font_size: "12sp"
                color: 0, 0, 0, 1
                size_hint_x: 0.7
                text_size: self.size
                halign: 'left'
                valign: 'center'
                padding: [10, 0]

            Button:
                text: "Notes"
                size_hint_x: 0.4
                background_color: 1, 0.3, 0.3, 1
                font_size: "16sp"
                on_press: root.show_notes()

            Button:
                text: "🚨 Emergency"
                size_hint_x: 0.3
                background_color: 1, 0.3, 0.3, 0.9
                font_size: "14sp"
                on_press: print('Emergency contacts notified')

        # Middle section with image and date
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.3
            padding: 15
            spacing: 8
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15, 15, 15, 15]

            AsyncImage:
                source: 'C:/Users/borut/Desktop/hackathon/app/1.jpg' 
                size_hint_y: 0.85
                allow_stretch: True
                keep_ratio: True
                canvas.before:
                    Color:
                        rgba: 0.95, 0.95, 0.95, 1
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [15]


            Label:
                text: "Today is Sunday, November 17, 2024"
                font_size: "16sp"
                color: 0.3, 0.3, 0.3, 1
                size_hint_y: 0.2

        # Bottom grid with main feature buttons
        GridLayout:
            cols: 2
            size_hint_y: 0.25
            spacing: 10
            padding: 5

            Button:
                text: "🧠\\nMemory\\nTraining"
                background_color: 0.6, 0.3, 1, 0.9
                on_press: root.manager.current = 'memory_training'
                font_size: "20sp"
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                padding: 10, 10
                canvas.before:
                    Color:
                        rgba: self.background_color
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [15]

            Button:
                text: "💊\\nMedication\\nReminder"
                background_color: 0.3, 0.7, 1, 1
                on_press: root.manager.current = 'medication'
                font_size: "18sp"
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                padding: 10, 10
                canvas.before:
                    Color:
                        rgba: self.background_color
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [10, 10, 10, 10]

            Button:
                text: "🤖\\nAI\\n"
                background_color: 0.3, 0.7, 1, 1
                on_press: root.manager.current = 'AI'
                font_size: "18sp"
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                padding: 10, 10

            Button:
                text: "📝\\nTo-Do\\nList"
                background_color: 0.3, 1, 0.5, 1
                on_press: root.manager.current = 'todo_list'
                font_size: "18sp"
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                padding: 10, 10
                canvas.before:
                    Color:
                        rgba: self.background_color
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [10, 10, 10, 10]

<MemoryTrainingScreen>:
    name: 'memory_training'
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10

        BoxLayout:
            size_hint_y: 0.1
            spacing: 10

            Button:
                text: "← Back"
                size_hint_x: 0.2
                on_press: root.manager.current = 'main'

            Label:
                text: "Memory Training Games"
                font_size: "20sp"
                color: 0, 0, 0, 1

        BoxLayout:
            id: game_container
            orientation: 'vertical'

<MedicationScreen>:
    name: 'medication'
    BoxLayout:
        orientation: 'vertical'
        padding: '15dp'
        spacing: '15dp'
        canvas.before:
            Color:
                rgba: 0.95, 0.95, 0.95, 1
            Rectangle:
                pos: self.pos
                size: self.size

        # Top Bar with Title and Back Button
        BoxLayout:
            size_hint_y: None
            height: '50dp'
            spacing: '10dp'
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15]

            Button:
                text: '← Back'
                size_hint_x: 0.2
                background_color: 0.3, 0.7, 0.9, 1
                on_press: root.manager.current = 'main'

            Label:
                text: 'Medication Manager'
                font_size: '22sp'
                bold: True
                color: 0.2, 0.2, 0.2, 1

        # Add New Medication Section
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: '180dp'
            padding: '15dp'
            spacing: '10dp'
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15]

            Label:
                text: 'Add New Medication'
                font_size: '18sp'
                color: 0.3, 0.3, 0.3, 1
                size_hint_y: None
                height: '30dp'
                bold: True

            TextInput:
                id: med_name_input
                hint_text: 'Medication Name'
                multiline: False
                size_hint_y: None
                height: '40dp'
                padding: '15dp', '10dp'
                background_color: 0.97, 0.97, 0.97, 1

            BoxLayout:
                size_hint_y: None
                height: '40dp'
                spacing: '10dp'

                TextInput:
                    id: med_time_input
                    hint_text: 'Time (HH:MM AM/PM)'
                    multiline: False
                    padding: '15dp', '10dp'
                    background_color: 0.97, 0.97, 0.97, 1

                Button:
                    text: '+ Add'
                    size_hint_x: 0.3
                    background_color: 0.3, 0.7, 0.9, 1
                    on_press: root.add_medication()
                    canvas.before:
                        Color:
                            rgba: self.background_color
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [10]

        # Medication List Section
        BoxLayout:
            orientation: 'vertical'
            padding: '15dp'
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15]

            Label:
                text: 'Current Medications'
                font_size: '18sp'
                color: 0.3, 0.3, 0.3, 1
                size_hint_y: None
                height: '30dp'
                bold: True

            ScrollView:
                do_scroll_x: False
                GridLayout:
                    id: medication_list
                    cols: 1
                    spacing: '10dp'
                    padding: '5dp'
                    size_hint_y: None
                    height: self.minimum_height

<TodoListScreen>:
    name: 'todo_list'
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        canvas.before:
            Color:
                rgba: 0.95, 0.95, 0.95, 1
            Rectangle:
                pos: self.pos
                size: self.size

        BoxLayout:
            size_hint_y: 0.1
            padding: 10
            spacing: 10
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15, 15, 15, 15]

            Button:
                text: "← Back"
                size_hint_x: 0.2
                on_press: root.manager.current = 'main'

            Label:
                text: "Daily Tasks"
                font_size: "20sp"
                color: 0, 0, 0, 1

        BoxLayout:
            size_hint_y: 0.1
            padding: 5
            spacing: 10

            TextInput:
                id: task_input
                hint_text: 'Add a new task...'
                multiline: False
                size_hint_x: 0.8
                on_text_validate: root.add_task()

            Button:
                text: '+'
                size_hint_x: 0.2
                on_press: root.add_task()

        ScrollView:
            BoxLayout:
                id: tasks_container
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: 10
                spacing: 5

<JogMemoryScreen>:
    name: 'AI'
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 15
        canvas.before:
            Color:
                rgba: 0.95, 0.95, 0.95, 1
            Rectangle:
                pos: self.pos
                size: self.size

        # Top Bar
        BoxLayout:
            size_hint_y: 0.1
            padding: 5
            spacing: 10
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15]

            Button:
                text: "← Back"
                size_hint_x: 0.2
                background_color: 0.3, 0.7, 0.9, 1
                on_press: root.manager.current = 'main'

            Label:
                text: "AI Assistant"
                font_size: "22sp"
                color: 0.2, 0.2, 0.2, 1
                bold: True

        # Main Content Area
        BoxLayout:
            orientation: 'vertical'
            padding: 15
            spacing: 10
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [15]

            Image:
                source: 'ai_assistant_icon.png'  # Add an appropriate icon
                size_hint: None, None
                size: "120dp", "120dp"
                pos_hint: {'center_x': 0.5}

            Label:
                text: "Your Personal Memory Assistant"
                font_size: "24sp"
                color: 0.2, 0.2, 0.2, 1
                size_hint_y: 0.2
                bold: True

            ScrollView:
                size_hint_y: 0.4
                Label:
                    text: "You can ask me about:\\n• Your personal information\\n• Daily schedule\\n• Memories and past events\\n• General knowledge\\n• And much more!"
                    font_size: "18sp"
                    color: 0.4, 0.4, 0.4, 1
                    size_hint_y: None
                    height: self.texture_size[1]
                    text_size: self.width, None
                    halign: 'center'

            Button:
                id: voice_button
                text: "Tap to Start Speaking"
                size_hint: None, None
                size: "250dp", "70dp"
                pos_hint: {'center_x': 0.5}
                background_color: 0.3, 0.7, 0.9, 1
                font_size: "20sp"
                on_press: root.activate_voice_assistant(self)
                canvas.before:
                    Color:
                        rgba: self.background_color
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [35]

            Label:
                id: status_label
                text: "Waiting to start..."
                font_size: "16sp"
                color: 0.5, 0.5, 0.5, 1
                size_hint_y: 0.1
'''

class LocationManager:
    def __init__(self, callback=None):
        self.callback = callback
        self.current_location = None
        self.is_mobile = self._check_mobile_device()
        
    def _check_mobile_device(self):
        try:
            gps.configure(on_location=lambda **kwargs: None)
            return True
        except Exception:
            return False
            
    def start_location_updates(self):
        if self.is_mobile:
            self._setup_gps()
        else:
            self._get_location_from_ip()
            
    def _setup_gps(self):
        try:
            gps.configure(on_location=self._on_location)
            gps.start(minTime=1000, minDistance=1)
        except Exception as e:
            print(f"GPS Error: {e}")
            self._get_location_from_ip()
            
    def _on_location(self, **kwargs):
        try:
            lat = kwargs.get('lat', None)
            lon = kwargs.get('lon', None)
            if lat and lon:
                self.current_location = (lat, lon)
                # Get city name using reverse geocoding
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(
                    f'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json',
                    headers=headers
                )
                data = response.json()
                city = data.get('address', {}).get('city', 
                       data.get('address', {}).get('town',
                       data.get('address', {}).get('village', 'Unknown City')))
                location_text = f"📍 {city} • {lat:.4f}, {lon:.4f}"
                
                if self.callback:
                    Clock.schedule_once(lambda dt: self.callback(location_text))
        except Exception as e:
            print(f"Location Error: {e}")
            if self.callback:
                Clock.schedule_once(lambda dt: self.callback("📍 Location Unavailable"))
                
    def _get_location_from_ip(self):
        try:
            response = requests.get('http://ipinfo.io/json')
            data = response.json()
            city = data.get('city', 'Unknown City')
            location = data.get('loc', '').split(',')
            
            if len(location) == 2:
                lat = float(location[0])
                lon = float(location[1])
                self.current_location = (lat, lon)
                location_text = f"📍 {city} • {lat:.4f}, {lon:.4f}"
                
                if self.callback:
                    Clock.schedule_once(lambda dt: self.callback(location_text))
            else:
                raise ValueError("Invalid location data")
        except Exception as e:
            print(f"IP Location Error: {e}")
            if self.callback:
                Clock.schedule_once(lambda dt: self.callback("📍 Location Unavailable"))
                
    def stop(self):
        if self.is_mobile:
            try:
                gps.stop()
            except:
                pass

# Update the MainScreen class:
class MainScreen(Screen):
    location_label = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location_manager = None
        
    def on_enter(self):
        App.get_running_app().voice_controller.start_listening()
        self.setup_location_manager()
        
    def on_leave(self):
        App.get_running_app().voice_controller.stop_listening()
        if self.location_manager:
            self.location_manager.stop()
            
    def setup_location_manager(self):
        self.location_manager = LocationManager(callback=self.update_location_label)
        self.location_manager.start_location_updates()
        
    def update_location_label(self, location_text):
        if hasattr(self, 'location_label') and self.location_label:
            self.location_label.text = location_text

    def show_notes(self):
        self.manager.current = 'notes'

class MemoryTrainingScreen(Screen):
    def on_enter(self):
        self.game = PatternMatchGame()
        self.ids.game_container.clear_widgets()
        self.ids.game_container.add_widget(self.game)

    def on_leave(self):
        self.ids.game_container.clear_widgets()

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from plyer import notification
import json
import os
from datetime import datetime

ALARM_SOUND_PATH = "alarm_sound.mp3"  # Path to alarm sound file


class MedicationItem(BoxLayout):
    def __init__(self, medication, delete_callback, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = '80dp'
        self.padding = '10dp'
        self.spacing = '10dp'

        # Background
        with self.canvas.before:
            Color(0.98, 0.98, 0.98, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[10])

        # Left side with icon
        left_box = BoxLayout(size_hint_x=0.15)
        with left_box.canvas.before:
            Color(0.3, 0.7, 0.9, 0.2)
            RoundedRectangle(pos=left_box.pos, size=left_box.size, radius=[10])

        icon_label = Label(
            text='💊',
            font_size='24sp'
        )
        left_box.add_widget(icon_label)

        # Middle section with medication details
        details_box = BoxLayout(
            orientation='vertical',
            size_hint_x=0.65,
            padding='5dp'
        )

        name_label = Label(
            text=medication['name'],
            font_size='18sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            halign='left'
        )
        name_label.bind(size=name_label.setter('text_size'))

        time_label = Label(
            text=f"🕒 {medication['time']}",
            font_size='16sp',
            color=(0.5, 0.5, 0.5, 1),
            halign='left'
        )
        time_label.bind(size=time_label.setter('text_size'))

        details_box.add_widget(name_label)
        details_box.add_widget(time_label)

        # Right side with controls
        controls_box = BoxLayout(
            size_hint_x=0.2,
            padding='5dp'
        )

        delete_btn = Button(
            text='×',
            font_size='20sp',
            background_color=(0.9, 0.3, 0.3, 1),
            size_hint=(None, None),
            size=('40dp', '40dp'),
            pos_hint={'center_y': 0.5}
        )
        delete_btn.bind(on_press=lambda x: delete_callback(medication))

        controls_box.add_widget(delete_btn)

        # Add all sections to main layout
        self.add_widget(left_box)
        self.add_widget(details_box)
        self.add_widget(controls_box)


class MedicationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.medications = []
        Clock.schedule_once(self.load_medications)
        Clock.schedule_interval(self.check_alarms, 60)

    def load_medications(self, dt=None):
        try:
            if os.path.exists('medications.json'):
                with open('medications.json', 'r') as f:
                    self.medications = json.load(f)
                self.update_medication_list()
        except Exception as e:
            print(f"Error loading medications: {e}")

    def add_medication(self):
        name = self.ids.med_name_input.text.strip()
        time = self.ids.med_time_input.text.strip()

        if name and time:
            # Validate time format
            try:
                datetime.strptime(time, '%I:%M %p')
                medication = {
                    'name': name,
                    'time': time,
                    'id': str(datetime.now().timestamp())
                }

                self.medications.append(medication)
                self.save_medications()
                self.update_medication_list()

                # Clear inputs
                self.ids.med_name_input.text = ''
                self.ids.med_time_input.text = ''

                # Show confirmation
                self.show_notification('Medication Added', f'{name} will remind at {time}')

            except ValueError:
                self.show_notification('Invalid Time', 'Please use format: HH:MM AM/PM')

    def update_medication_list(self):
        medication_list = self.ids.medication_list
        medication_list.clear_widgets()

        # Sort medications by time
        sorted_medications = sorted(
            self.medications,
            key=lambda x: datetime.strptime(x['time'], '%I:%M %p')
        )

        for med in sorted_medications:
            item = MedicationItem(
                medication=med,
                delete_callback=self.delete_medication
            )
            medication_list.add_widget(item)

    def delete_medication(self, medication):
        self.medications.remove(medication)
        self.save_medications()
        self.update_medication_list()
        self.show_notification('Medication Removed', f'{medication["name"]} has been removed')

    def save_medications(self):
        try:
            with open('medications.json', 'w') as f:
                json.dump(self.medications, f)
        except Exception as e:
            print(f"Error saving medications: {e}")

    def show_notification(self, title, message):
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=5
            )
        except Exception as e:
            print(f"Notification error: {e}")

    def check_alarms(self, dt):
        current_time = datetime.now().strftime('%I:%M %p')
        for med in self.medications:
            if med['time'] == current_time:
                self.trigger_alarm(med)

    def trigger_alarm(self, medication):
        # Play sound alarm
        try:
            sound = SoundLoader.load(ALARM_SOUND_PATH)
            if sound:
                sound.play()
                Clock.schedule_once(lambda dt: sound.stop(), 20)
        except Exception as e:
            print(f"Error playing alarm: {e}")

        # Show notification
        self.show_notification(
            'Medication Reminder',
            f'Time to take {medication["name"]}!'
        )

class JogMemoryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voice_assistant_active = False
        
        # Pre-download the NLTK data (this happens only once when the screen is initialized)
        self.ensure_nltk_data()

    def ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def activate_voice_assistant(self, button):
        if not self.voice_assistant_active:
            self.voice_assistant_active = True
            button.text = "Stop Voice Assistant"
            button.background_color = [1, 0.3, 0.3, 1]  # Red color when active
            # Start the voice assistant in a separate thread
            Thread(target=self.run_voice_assistant_thread, daemon=True).start()
        else:
            self.voice_assistant_active = False
            button.text = "Activate Voice Assistant"
            button.background_color = [0.3, 0.7, 0.9, 1]  # Original blue color

    def run_voice_assistant_thread(self):
        # Initialize and run the voice assistant
        run_voice_assistant()


class TodoListScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = []
        Clock.schedule_once(self.setup_initial_tasks)

    def setup_initial_tasks(self, dt):
        initial_tasks = [
            "Take morning medication",
            "Do 15 minutes of memory exercises",
            "Walk for 30 minutes",
            "Call family member",
            "Read for 20 minutes"
        ]
        for task in initial_tasks:
            self.add_task_widget(task)

    def add_task(self):
        task_input = self.ids.task_input
        if task_input.text.strip():
            self.add_task_widget(task_input.text)
            task_input.text = ''

    def add_task_widget(self, task_text):
        task_widget = TaskItem(task_text=task_text)
        self.ids.tasks_container.add_widget(task_widget)
        self.tasks.append(task_widget)

    def toggle_task(self, task_item):
        task_item.completed = not task_item.completed
        self.ids.tasks_container.remove_widget(task_item)
        
        incomplete_tasks = [t for t in self.tasks if not t.completed]
        complete_tasks = [t for t in self.tasks if t.completed]
        
        self.ids.tasks_container.clear_widgets()
        
        for task in incomplete_tasks:
            self.ids.tasks_container.add_widget(task)
        for task in complete_tasks:
            self.ids.tasks_container.add_widget(task)

class VoiceController:
    def __init__(self, screen_manager=None):
        self.screen_manager = screen_manager
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Command dictionary with variations
        self.commands = {
            # Memory Training commands
            "memory training": "memory_training",
            "memory games": "memory_training",
            "memory": "memory_training",
            "games": "memory_training",
            "brain games": "memory_training",
            
            # Medication commands
            "medication": "medication",
            "medicines": "medication",
            "pills": "medication",
            "reminders": "medication",
            
            # AI Assistant commands
            "ai": "AI",
            "assistant": "AI",
            "help": "AI",
            "chat": "AI",
            
            # Todo List commands
            "todo": "todo_list",
            "tasks": "todo_list",
            "list": "todo_list",
            "to do": "todo_list",
            
            # Home screen commands
            "home": "main",
            "main": "main",
            "back": "main"
        }
        
        # Initialize text-to-speech engine
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)

    def set_screen_manager(self, screen_manager):
        self.screen_manager = screen_manager

    def start_listening(self):
        if not self.is_listening:
            self.is_listening = True
            Thread(target=self._listen_loop, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False

    def _listen_loop(self):
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    print("\nListening...")
                    # Reduce ambient noise adjustment duration
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Set shorter timeout and phrase time limit
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    try:
                        print("Processing speech...")
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Recognized: '{text}'")
                        
                        # Process the command
                        if self._process_command(text):
                            print("Command executed successfully")
                        else:
                            print("No matching command found")
                            
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results: {e}")
                        continue
                        
            except sr.WaitTimeoutError:
                print("Listening timeout - continuing...")
                continue
            except Exception as e:
                print(f"Error in listen loop: {str(e)}")
                time.sleep(1)
                continue

    def _process_command(self, text):
        try:
            # Check for navigation commands
            for command_phrase, screen_name in self.commands.items():
                if command_phrase in text:
                    print(f"Navigating to: {screen_name}")
                    self.speak(f"Going to {command_phrase}")
                    Clock.schedule_once(lambda dt: self._navigate_to_screen(screen_name))
                    return True
                    
            # Handle special commands or conversation
            if self.screen_manager.current == "AI":
                # Process as conversation in AI screen
                print("Processing as conversation in AI screen")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error processing command: {str(e)}")
            return False

    def _navigate_to_screen(self, screen_name):
        try:
            if self.screen_manager and screen_name in self.screen_manager.screen_names:
                self.screen_manager.current = screen_name
            else:
                print(f"Screen '{screen_name}' not found")
        except Exception as e:
            print(f"Navigation error: {str(e)}")

    def speak(self, text):
        try:
            print(f"Assistant: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {str(e)}")

    def process_ai_conversation(self, text):
        # Add conversation processing logic here
        response = "I understood your message: " + text
        self.speak(response)

class NotesScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = JsonStore('notes.json')
        self.setup_ui()
        
    def setup_ui(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Top bar with back button and save button
        top_bar = BoxLayout(size_hint_y=0.1, spacing=10)
        back_button = Button(
            text="← Back",
            size_hint=(None, None),
            size=("100dp", "40dp"),
            on_press=lambda x: self.go_back()
        )
        save_button = Button(
            text="Save",
            size_hint=(None, None),
            size=("100dp", "40dp"),
            on_press=lambda x: self.save_note()
        )
        title_label = Label(
            text="Notes",
            font_size="24sp",
            bold=True,
            size_hint_x=0.6
        )
        
        top_bar.add_widget(back_button)
        top_bar.add_widget(title_label)
        top_bar.add_widget(save_button)
        
        # Text input for notes
        self.notes_input = TextInput(
            multiline=True,
            size_hint=(1, 0.9),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            padding=(10, 10),
            font_size='16sp'
        )
        
        layout.add_widget(top_bar)
        layout.add_widget(self.notes_input)
        
        self.add_widget(layout)
    
    def on_enter(self):
        # Load last saved note when entering the screen
        try:
            if self.store.exists('last_note'):
                last_note = self.store.get('last_note')
                self.notes_input.text = last_note['content']
        except Exception as e:
            print(f"Error loading note: {e}")
    
    def save_note(self):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.store.put('last_note',
                content=self.notes_input.text,
                timestamp=timestamp
            )
            # Show a brief confirmation
            self.show_save_confirmation()
        except Exception as e:
            print(f"Error saving note: {e}")
    
    def show_save_confirmation(self):
        popup = Popup(
            title='Saved',
            content=Label(text='Note saved successfully!'),
            size_hint=(None, None),
            size=(200, 100)
        )
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 1)
    
    def go_back(self):
        self.save_note()  # Auto-save when going back
        self.manager.current = 'main'


class MemoryBoxApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voice_controller = VoiceController()
        
    def build(self):
        Builder.load_string(KV)
        sm = ScreenManager()
        
        # Add all screens
        sm.add_widget(MainScreen())
        sm.add_widget(MemoryTrainingScreen())
        sm.add_widget(MedicationScreen())
        sm.add_widget(JogMemoryScreen())
        sm.add_widget(TodoListScreen())
        sm.add_widget(NotesScreen(name='notes'))  # Add the new NotesScreen

        self.voice_controller.set_screen_manager(sm)
        
        return sm
        
    def on_stop(self):
        self.voice_controller.stop_listening()
        # Stop location updates
        main_screen = self.root.get_screen('main')
        if main_screen.location_manager:
            main_screen.location_manager.stop()

if __name__ == "__main__":
    MemoryBoxApp().run()