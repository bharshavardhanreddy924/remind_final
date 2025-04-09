try:
    from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
    from werkzeug.security import generate_password_hash, check_password_hash
    from datetime import datetime, timedelta
    from bson.objectid import ObjectId
    import os
    import random
    import json
    import nltk
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import re
    import certifi
    from flask import g
    from groq import Groq
    import logging
    from functools import wraps  # Add this import for the authentication decorator
    
    # New imports for face recognition and reminiscence therapy
    import cv2
    from PIL import Image
    import face_recognition
    import glob
    import shutil
    from werkzeug.utils import secure_filename
    import torch
    from pathlib import Path
    import uuid
    import pickle
    import time
    import requests  # For making HTTP requests
except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Please make sure all required packages are installed by running: pip install -r requirements.txt")
    exit(1)

from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'memorycareappsecretkey')
app.permanent_session_lifetime = timedelta(days=7)

# Set secure headers for PWA
@app.after_request
def add_pwa_headers(response):
    # Add headers to help with PWA functionality
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    response.headers['Permissions-Policy'] = 'geolocation=(self), microphone=(), camera=()'
    return response

# Offline fallback page
@app.route('/offline')
def offline():
    return render_template('offline.html')

# MongoDB Configuration
from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://bharshavardhanreddy924:516474Ta@data-dine.5oghq.mongodb.net/?retryWrites=true&w=majority&ssl=true"

try:
    client = MongoClient(uri, tlsCAFile=certifi.where())
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("✅ Pinged your deployment. You successfully connected to MongoDB!")
    db = client['memorycare_db']
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None  # Prevent application crashes

# File Upload Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_extensions=None):
    """Check if file has an allowed extension"""
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def resize_image_if_needed(image_path, max_size=(1024, 1024), quality=85):
    """Resize an image if it's too large"""
    try:
        img = Image.open(image_path)
        
        # Only resize if the image is larger than max_size
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)
            img.save(image_path, optimize=True, quality=quality)
            print(f"Resized image at {image_path}")
        
        return True
    except Exception as e:
        print(f"Error resizing image: {e}")
        return False

# Database initialization
def init_db():
    # Create initial collections if they don't exist
    if db is not None:
        # Check if users collection has any documents
        if db.users.count_documents({}) == 0:
            print("Initializing database with default collections...")
            
            # Create initial admin user if no users exist
            db.users.insert_one({
                'name': 'Admin User',
                'email': 'admin@memorycare.com',
                'password': generate_password_hash('admin123'),
                'user_type': 'caretaker',
                'created_at': datetime.now(),
                'personal_info': 'Administrator account'
            })
            print("Created admin user")

# Initialize database
if db is not None:
    init_db()
    
    # Create locations collection if it doesn't exist
    if 'locations' not in db.list_collection_names():
        db.create_collection('locations')
        print("Created locations collection")

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Helper functions
def check_db_connection():
    """Check if MongoDB connection is available and flash an error message if not"""
    if db is None:
        flash('Database connection error. Please try again later.', 'danger')
        return False
    return True

def get_user_data(user_id):
    if not check_db_connection():
        return None
    return db.users.find_one({"_id": ObjectId(user_id)})

def get_caretaker_patients(caretaker_id):
    if not check_db_connection():
        return []
    return list(db.users.find({"caretaker_id": ObjectId(caretaker_id)}))

class PersonalInfoQA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
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
        
    def train(self, text):
        sentences = sent_tokenize(text)
        cleaned_sentences = [sent.strip() for sent in sentences if sent.strip()]
        self.knowledge_base = cleaned_sentences
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_sentences)
    
    def generate_response(self, prompt):
        if not self.knowledge_base:
            return "I haven't been trained with any memories yet."
        
        try:
            cleaned_prompt = prompt.strip().lower()
            prompt_vector = self.vectorizer.transform([cleaned_prompt])
            
            similarities = cosine_similarity(prompt_vector, self.tfidf_matrix).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            
            relevant_sentences = [self.knowledge_base[idx] for idx in sorted_indices[:3] if similarities[idx] > 0.05]
            
            if relevant_sentences:
                starter = random.choice(self.conversation_starters)
                response = " ".join(relevant_sentences)
                return f"{starter} {response}"
            else:
                keywords = {
                    'hobbies': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'hobby': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'interests': 'I enjoy several hobbies like hiking, photography, and playing sports.',
                    'interest': 'I enjoy several hobbies like hiking, photography, and playing sports.',
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

# Routes
@app.route('/splash')
def splash():
    """Serve the splash screen for PWA startup"""
    return render_template('splash.html')

@app.route('/')
def index():
    """Main entry point which can handle standalone parameter"""
    # Check if app is running in standalone mode
    standalone = request.args.get('standalone') == 'true'
    
    # Add standalone parameter to session if provided
    if standalone:
        session['standalone'] = True
    
    # Check if user is logged in, if so redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    # Otherwise show the login page
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check if MongoDB connection is available
        if not check_db_connection():
            return render_template('login.html')
            
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = db.users.find_one({"email": email})
        
        if user and check_password_hash(user['password'], password):
            session.permanent = True
            session['user_id'] = str(user['_id'])
            session['user_type'] = user['user_type']
            session['name'] = user['name']
            
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Check if MongoDB connection is available
        if not check_db_connection():
            return render_template('register.html')
            
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        
        # Check if email already exists
        if db.users.find_one({"email": email}):
            flash('Email already exists', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "user_type": user_type,
            "created_at": datetime.now(),
            "personal_info": "I am a person who needs memory care assistance."
        }
        
        # For patient-type users, allow caretaker assignment
        if user_type == "user" and request.form.get('caretaker_email'):
            caretaker = db.users.find_one({"email": request.form.get('caretaker_email'), "user_type": "caretaker"})
            if caretaker:
                new_user["caretaker_id"] = caretaker['_id']
            else:
                flash('Caretaker email not found', 'warning')
        
        user_id = db.users.insert_one(new_user).inserted_id
        
        # Initialize collections for the user
        db.tasks.insert_one({
            "user_id": user_id,
            "tasks": [
                {"text": "Take morning medication", "completed": False},
                {"text": "Do 15 minutes of memory exercises", "completed": False},
                {"text": "Walk for 30 minutes", "completed": False},
                {"text": "Call family member", "completed": False},
                {"text": "Read for 20 minutes", "completed": False}
            ]
        })
        
        db.medications.insert_one({
            "user_id": user_id,
            "medications": []
        })
        
        db.notes.insert_one({
            "user_id": user_id,
            "content": "",
            "updated_at": datetime.now()
        })
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check MongoDB connection
    if not check_db_connection():
        return redirect(url_for('logout'))
    
    user_id = session['user_id']
    user_type = session['user_type']
    
    if user_type == 'caretaker':
        patients = get_caretaker_patients(ObjectId(user_id))
        return render_template('dashboard.html', patients=patients)
    else:
        user = get_user_data(ObjectId(user_id))
        if user is None:
            flash('User data not found. Please try logging in again.', 'danger')
            return redirect(url_for('logout'))
        
        # Get user's task data
        task_data = db.tasks.find_one({"user_id": ObjectId(user_id)})
        if not task_data:
            task_data = {"tasks": []}
        
        # Get medication data
        med_data = db.medications.find_one({"user_id": ObjectId(user_id)})
        if not med_data:
            med_data = {"medications": []}
        
        # Current date info
        today = datetime.now()
        formatted_date = today.strftime("%A, %B %d, %Y")
        
        return render_template('dashboard.html', 
                               user=user, 
                               tasks=task_data['tasks'], 
                               medications=med_data['medications'],
                               date=formatted_date)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/tasks', methods=['GET', 'POST'])
def tasks():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    
    if request.method == 'POST':
        if request.form.get('action') == 'add':
            task_text = request.form.get('task_text')
            
            # Add task to database
            db.tasks.update_one(
                {"user_id": user_id},
                {"$push": {"tasks": {"text": task_text, "completed": False}}}
            )
            
            # Schedule notification for the new task
            if schedule_task_notification(task_text, user_id):
                flash('Task added successfully with notification scheduled', 'success')
            else:
                flash('Task added successfully but notification scheduling failed', 'warning')
        
        elif request.form.get('action') == 'update':
            task_index = int(request.form.get('task_index'))
            task_status = 'completed' in request.form
            
            # Get existing tasks
            task_data = db.tasks.find_one({"user_id": user_id})
            tasks = task_data.get('tasks', [])
            
            # Update specific task status
            if 0 <= task_index < len(tasks):
                tasks[task_index]['completed'] = task_status
                
                # Update in database
                db.tasks.update_one(
                    {"user_id": user_id},
                    {"$set": {"tasks": tasks}}
                )
                flash('Task updated', 'success')
        
        elif request.form.get('action') == 'delete':
            task_index = int(request.form.get('task_index'))
            
            # Get existing tasks
            task_data = db.tasks.find_one({"user_id": user_id})
            tasks = task_data.get('tasks', [])
            
            # Remove specific task
            if 0 <= task_index < len(tasks):
                tasks.pop(task_index)
                
                # Update in database
                db.tasks.update_one(
                    {"user_id": user_id},
                    {"$set": {"tasks": tasks}}
                )
                flash('Task deleted', 'success')
    
    # Get updated task list
    task_data = db.tasks.find_one({"user_id": user_id})
    if not task_data:
        tasks = []
    else:
        tasks = task_data.get('tasks', [])
    
    return render_template('tasks.html', tasks=tasks)

@app.route('/medications', methods=['GET', 'POST'])
def medications():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    
    if request.method == 'POST':
        if request.form.get('action') == 'add':
            med_name = request.form.get('med_name')
            med_time = request.form.get('med_time')
            
            if med_name and med_time:
                # Validate time format
                try:
                    # Try to parse the time to validate format
                    time_obj = datetime.strptime(med_time, '%I:%M %p')
                    
                    new_med = {
                        "id": str(datetime.now().timestamp()),
                        "name": med_name,
                        "time": med_time,
                    }
                    
                    # Add medication to database
                    db.medications.update_one(
                        {"user_id": user_id},
                        {"$push": {"medications": new_med}},
                        upsert=True
                    )
                    
                    # Schedule notification for the new medication
                    if schedule_medication_notification(med_name, med_time, user_id):
                        flash('Medication added successfully with notification scheduled', 'success')
                    else:
                        flash('Medication added successfully but notification scheduling failed', 'warning')
                        
                except ValueError:
                    flash('Invalid time format. Please use HH:MM AM/PM', 'danger')
        
        elif request.form.get('action') == 'delete':
            med_id = request.form.get('med_id')
            
            # Remove medication from database
            db.medications.update_one(
                {"user_id": user_id},
                {"$pull": {"medications": {"id": med_id}}}
            )
            
            # Remove any scheduled notifications for this medication
            db.notifications.delete_many({
                "user_id": user_id,
                "type": "medication",
                "medication_id": med_id
            })
            
            flash('Medication deleted', 'success')
    
    # Get updated medication list
    med_data = db.medications.find_one({"user_id": user_id})
    if not med_data:
        medications = []
    else:
        medications = med_data.get('medications', [])
        
        # Sort medications by time
        def time_key(med):
            try:
                return datetime.strptime(med['time'], '%I:%M %p').time()
            except:
                return datetime.min.time()
                
        medications = sorted(medications, key=time_key)
    
    return render_template('medications.html', medications=medications)

@app.route('/memory_training')
def memory_training():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('memory_training.html')

@app.route('/notes', methods=['GET', 'POST'])
def notes():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    
    if request.method == 'POST':
        content = request.form.get('content')
        
        db.notes.update_one(
            {"user_id": user_id},
            {"$set": {"content": content, "updated_at": datetime.now()}},
            upsert=True
        )
        
        flash('Notes saved successfully', 'success')
    
    # Get user's notes
    notes_data = db.notes.find_one({"user_id": user_id})
    if not notes_data:
        content = ""
    else:
        content = notes_data.get('content', "")
    
    return render_template('notes.html', content=content)

@app.route('/ai_assistant')
def ai_assistant():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    user = get_user_data(ObjectId(user_id))
    
    return render_template('ai_assistant.html', user=user)

@app.route('/api/ai_response', methods=['POST'])
def ai_response():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = ObjectId(session['user_id'])
    user = get_user_data(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    
    # Initialize QA system with user's personal info
    qa_system = PersonalInfoQA()
    qa_system.train(user.get('personal_info', ''))
    
    # Generate response
    response = qa_system.generate_response(prompt)
    
    return jsonify({"response": response})

@app.route('/manage_patient/<patient_id>')
def manage_patient(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        patient = get_user_data(patient_id_obj)
        
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('dashboard'))
        
        # Get patient's task data
        task_data = db.tasks.find_one({"user_id": patient_id_obj})
        if not task_data:
            tasks = []
        else:
            tasks = task_data.get('tasks', [])
        
        # Get medication data
        med_data = db.medications.find_one({"user_id": patient_id_obj})
        if not med_data:
            medications = []
        else:
            medications = med_data.get('medications', [])
            
            # Sort medications by time
            def time_key(med):
                try:
                    return datetime.strptime(med['time'], '%I:%M %p').time()
                except:
                    return datetime.min.time()
                    
            medications = sorted(medications, key=time_key)
        
        # Get notes data
        notes_data = db.notes.find_one({"user_id": patient_id_obj})
        if not notes_data:
            notes_content = ""
        else:
            notes_content = notes_data.get('content', "")
        
        return render_template('manage_patient.html', 
                              patient=patient,
                              tasks=tasks,
                              medications=medications,
                              notes_content=notes_content)
                              
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/update_patient_info/<patient_id>', methods=['POST'])
def update_patient_info(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        personal_info = request.form.get('personal_info', '')
        
        db.users.update_one(
            {"_id": patient_id_obj},
            {"$set": {"personal_info": personal_info}}
        )
        
        flash('Patient information updated successfully', 'success')
        return redirect(url_for('manage_patient', patient_id=patient_id))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/add_patient_task/<patient_id>', methods=['POST'])
def add_patient_task(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        task_text = request.form.get('task_text')
        
        db.tasks.update_one(
            {"user_id": patient_id_obj},
            {"$push": {"tasks": {"text": task_text, "completed": False}}},
            upsert=True
        )
        
        flash('Task added successfully', 'success')
        return redirect(url_for('manage_patient', patient_id=patient_id))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('manage_patient', patient_id=patient_id))

@app.route('/add_patient_medication/<patient_id>', methods=['POST'])
def add_patient_medication(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        med_name = request.form.get('med_name')
        med_time = request.form.get('med_time')
        
        if med_name and med_time:
            # Validate time format
            try:
                # Try to parse the time to validate format
                time_obj = datetime.strptime(med_time, '%I:%M %p')
                
                new_med = {
                    "id": str(datetime.now().timestamp()),
                    "name": med_name,
                    "time": med_time,
                }
                
                db.medications.update_one(
                    {"user_id": patient_id_obj},
                    {"$push": {"medications": new_med}},
                    upsert=True
                )
                
                flash('Medication added successfully', 'success')
            except ValueError:
                flash('Invalid time format. Please use HH:MM AM/PM', 'danger')
        
        return redirect(url_for('manage_patient', patient_id=patient_id))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('manage_patient', patient_id=patient_id))

@app.route('/update_patient_notes/<patient_id>', methods=['POST'])
def update_patient_notes(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        content = request.form.get('content')
        
        db.notes.update_one(
            {"user_id": patient_id_obj},
            {"$set": {"content": content, "updated_at": datetime.now()}},
            upsert=True
        )
        
        flash('Notes updated successfully', 'success')
        return redirect(url_for('manage_patient', patient_id=patient_id))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('manage_patient', patient_id=patient_id))

# Location tracking routes
@app.route('/api/update_location', methods=['POST'])
def update_location():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        # Hardcoded location for St. Joseph College of Engineering
        latitude = 12.8699
        longitude = 80.2184
        accuracy = 0.0
        source = "hardcoded"
        timestamp = datetime.now()
        address = "St. Joseph College of Engineering, Chennai, Tamil Nadu, India"
        
        # Get existing location data
        existing_location = db.locations.find_one({"user_id": ObjectId(session['user_id'])})
        
        # Update location in database with source tracking
        location_data = {
            "user_id": ObjectId(session['user_id']),
            "latitude": latitude,
            "longitude": longitude,
            "accuracy": accuracy,
            "source": source,
            "timestamp": timestamp,
            "address": address,
            "location_history": []
        }
        
        # Add to location history if it exists
        if existing_location:
            location_data["location_history"] = existing_location.get("location_history", [])
            # Keep only last 10 locations
            location_data["location_history"].append({
                "latitude": existing_location["latitude"],
                "longitude": existing_location["longitude"],
                "accuracy": existing_location.get("accuracy", 0),
                "source": existing_location.get("source", "unknown"),
                "timestamp": existing_location.get("timestamp", datetime.now())
            })
            if len(location_data["location_history"]) > 10:
                location_data["location_history"].pop(0)
        
        # Update location in database
        db.locations.update_one(
            {"user_id": ObjectId(session['user_id'])},
            {"$set": location_data},
            upsert=True
        )
        
        return jsonify({
            "status": "success",
            "message": f"Location updated from {source}",
            "accuracy": accuracy
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/patient_location/<patient_id>')
def patient_location(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return redirect(url_for('login'))
    
    try:
        patient_id_obj = ObjectId(patient_id)
        patient = get_user_data(patient_id_obj)
        
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('dashboard'))
        
        # Get patient's location data
        location_data = db.locations.find_one({"user_id": patient_id_obj})
        
        return render_template('patient_location.html',
                             patient=patient,
                             location=location_data,
                             api_key='AIzaSyBuygAgUK0_5q7aHidZuWH7AKSJmdU75RY')
                             
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/api/get_patient_location/<patient_id>')
def get_patient_location(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        patient_id_obj = ObjectId(patient_id)
        location_data = db.locations.find_one({"user_id": patient_id_obj})
        
        if location_data:
            return jsonify({
                "latitude": location_data['latitude'],
                "longitude": location_data['longitude'],
                "accuracy": location_data.get('accuracy', 0),
                "address": location_data.get('address', ''),
                "timestamp": location_data['timestamp'].isoformat()
            })
        else:
            return jsonify({"error": "No location data found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Internal server error. Please try again later."), 500

@app.route('/db_status')
def db_status():
    """Route to check database connection status - admin use only"""
    try:
        if db is None:
            return jsonify({"status": "error", "message": "MongoDB connection not established"})
        
        # Try to ping the database
        db.command('ping')
        return jsonify({
            "status": "connected", 
            "message": "MongoDB connection is working", 
            "collections": db.list_collection_names()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Add session middleware to detect standalone mode
@app.before_request
def handle_standalone_mode():
    """Check and propagate standalone mode flag"""
    # Check if we have standalone in the session
    if session.get('standalone'):
        # Add it to g object for templates
        g.standalone = True
    
    # Also check query parameter
    if request.args.get('standalone') == 'true':
        session['standalone'] = True
        g.standalone = True

# Add these helper functions after the existing helper functions
def schedule_task_notification(task_text, user_id):
    """Schedule a notification for a new task"""
    try:
        # Get user's notification preferences
        user = get_user_data(ObjectId(user_id))
        if not user or not user.get('notifications_enabled', True):
            return

        # Schedule notification 20 minutes before the task time
        notification_data = {
            "type": "task",
            "title": "Task Reminder",
            "body": f"Task: {task_text}",
            "user_id": user_id,
            "scheduled_time": datetime.now() + timedelta(minutes=20)
        }
        
        db.notifications.insert_one(notification_data)
        return True
    except Exception as e:
        print(f"Error scheduling task notification: {e}")
        return False

def schedule_medication_notification(med_name, med_time, user_id):
    """Schedule a notification for a new medication"""
    try:
        # Get user's notification preferences
        user = get_user_data(ObjectId(user_id))
        if not user or not user.get('notifications_enabled', True):
            return

        # Convert medication time to datetime
        try:
            med_datetime = datetime.strptime(med_time, '%I:%M %p')
            # Set the time to today's date
            med_datetime = med_datetime.replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            )
            
            # If the time has already passed today, schedule for tomorrow
            if med_datetime < datetime.now():
                med_datetime += timedelta(days=1)
            
            # Schedule notification 20 minutes before medication time
            notification_time = med_datetime - timedelta(minutes=20)
            
            notification_data = {
                "type": "medication",
                "title": "Medication Reminder",
                "body": f"Time to take: {med_name}",
                "user_id": user_id,
                "scheduled_time": notification_time,
                "medication_time": med_datetime
            }
            
            db.notifications.insert_one(notification_data)
            return True
        except ValueError:
            print(f"Invalid time format: {med_time}")
            return False
    except Exception as e:
        print(f"Error scheduling medication notification: {e}")
        return False

# Face Recognition and Reminiscence Therapy Features
# Define folder paths
PEOPLE_FOLDER = os.path.join('static', 'uploads', 'people')
UPLOAD_FOLDER = os.path.join('static', 'uploads', 'uploaded')
OLD_IMAGES_FOLDER = os.path.join('static', 'uploads', 'images_old')
REFERENCES_FOLDER = os.path.join('static', 'uploads', 'references')

# Ensure all folders exist
os.makedirs(PEOPLE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OLD_IMAGES_FOLDER, exist_ok=True)
os.makedirs(REFERENCES_FOLDER, exist_ok=True)

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_face_encodings_from_folder():
    """Load face encodings from the people folder with improved efficiency"""
    known_face_encodings = []
    known_face_names = []
    known_face_relations = []
    
    # Cache file for encodings to speed up loading
    cache_file = os.path.join(PEOPLE_FOLDER, "encodings_cache.pkl")
    
    # Check if cache exists and is newer than all image files
    use_cache = False
    if os.path.exists(cache_file):
        cache_mtime = os.path.getmtime(cache_file)
        all_images_older = True
        
        # Get all image files from the people folder
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(PEOPLE_FOLDER, f"*.{ext}")))
        
        # Check if any image is newer than cache
        for img_file in image_files:
            if os.path.getmtime(img_file) > cache_mtime:
                all_images_older = False
                break
        
        use_cache = all_images_older and len(image_files) > 0
    
    # Use cache if valid
    if use_cache:
        try:
            print("Loading face encodings from cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                known_face_encodings = cache_data['encodings']
                known_face_names = cache_data['names']
                known_face_relations = cache_data['relations']
                print(f"Loaded {len(known_face_encodings)} faces from cache")
                return known_face_encodings, known_face_names, known_face_relations
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Continue with normal loading if cache fails
    
    # Get all image files from the people folder
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(PEOPLE_FOLDER, f"*.{ext}")))
    
    print(f"Loading {len(image_files)} reference face images...")
    
    for image_file in image_files:
        # Extract name and relation from the filename
        base_name = os.path.basename(image_file)
        name_relation = os.path.splitext(base_name)[0]  # Remove the extension
        
        try:
            # Check if the filename has the format Name_Relation
            if '_' in name_relation:
                name, relation = name_relation.split('_', 1)
                
                # Load and encode the face using OpenCV for faster loading
                image = cv2.imread(image_file)
                if image is None:
                    print(f"Could not load image: {image_file}")
                    continue
                    
                # Convert to RGB for face_recognition
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face locations first
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                if face_locations:
                    # Get encodings for the faces
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    
                    if face_encodings:
                        # Store the encoding (use first face if multiple detected)
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(name)
                        known_face_relations.append(relation)
                        print(f"Added reference face: {name} ({relation})")
                    else:
                        print(f"No encodings found in {image_file}")
                else:
                    print(f"No face detected in {image_file}")
            else:
                print(f"Skipping {image_file}: Filename not in Name_Relation format")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Save to cache for future use
    if known_face_encodings:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': known_face_encodings,
                    'names': known_face_names,
                    'relations': known_face_relations
                }, f)
            print(f"Saved {len(known_face_encodings)} face encodings to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    return known_face_encodings, known_face_names, known_face_relations

def recognize_faces_in_image(image_path):
    """
    Recognizes faces in an image by comparing with reference faces.
    
    Args:
        image_path (str): Path to the image file to analyze
        
    Returns:
        list: List of dictionaries with recognized face information or
        dict: Error information if processing fails
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Validate image format
        try:
            img = Image.open(image_path)
            
            # Save a temporary copy of the image if it's PNG, GIF, or WEBP 
            # (to ensure compatibility with face_recognition library)
            temp_image = None
            if img.format in ['PNG', 'GIF', 'WEBP']:
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB for better compatibility
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
                    img = rgb_img
                
                temp_path = f"{os.path.splitext(image_path)[0]}_temp.jpg"
                img.save(temp_path, 'JPEG', quality=95)
                image_path = temp_path
                temp_image = temp_path
            
        except Exception as e:
            return {"error": f"Invalid image format: {str(e)}"}
        
        # Process image with face recognition
        try:
            # Attempt to load the image
            try:
                image = face_recognition.load_image_file(image_path)
            except Exception as e:
                return {"error": f"Failed to load image: {str(e)}"}
            
            # Resize large images for faster processing
            height, width, _ = image.shape
            max_size = 1000
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # First try HOG (faster) method for face detection
            face_locations = face_recognition.face_locations(image, model="hog")
            
            # If no faces found, try CNN method (more accurate but slower)
            if not face_locations and os.path.getsize(image_path) < 5 * 1024 * 1024:  # < 5MB to avoid memory issues
                try:
                    app.logger.info("No faces found with HOG, trying CNN model")
                    face_locations = face_recognition.face_locations(image, model="cnn")
                except Exception as e:
                    app.logger.warning(f"CNN face detection failed: {str(e)}")
            
            if not face_locations:
                if temp_image and os.path.exists(temp_image):
                    os.remove(temp_image)
                return {"error": "No faces found in the image"}
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                if temp_image and os.path.exists(temp_image):
                    os.remove(temp_image)
                return {"error": "Could not encode faces in the image"}
            
            # Get reference faces from database
            reference_faces = get_reference_faces()
            
            # Check if we have any reference faces to compare against
            if not reference_faces:
                if temp_image and os.path.exists(temp_image):
                    os.remove(temp_image)
                return []  # Return empty list to indicate no matches (but no error)
            
            # Extract names and encodings from reference data
            ref_names = [face.get('name', 'Unknown') for face in reference_faces]
            ref_relationships = [face.get('relationship', '') for face in reference_faces]
            ref_encodings = []
            
            for face in reference_faces:
                # Convert string encoding back to numpy array if needed
                encoding = face.get('encoding')
                if encoding and isinstance(encoding, list):
                    ref_encodings.append(np.array(encoding))
                else:
                    app.logger.warning(f"Missing or invalid encoding for reference face: {face.get('name')}")
            
            # Initialize results list
            results = []
            
            # Compare each detected face with reference faces
            for i, face_encoding in enumerate(face_encodings):
                try:
                    if not ref_encodings:
                        continue
                    
                    # Compare the face with all reference faces
                    matches = face_recognition.compare_faces(ref_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(ref_encodings, face_encoding)
                    
                    if True in matches:
                        # Find the best match
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        if matches[best_match_index]:
                            name = ref_names[best_match_index]
                            relationship = ref_relationships[best_match_index]
                            
                            # Add to results if confidence is above threshold
                            if confidence >= 0.5:  # 50% confidence threshold
                                results.append({
                                    'name': name,
                                    'relationship': relationship,
                                    'confidence': float(confidence),
                                    'face_location': face_locations[i]
                                })
                except Exception as e:
                    app.logger.error(f"Error matching face {i}: {str(e)}")
                    continue  # Continue with next face
            
            # Cleanup temporary file if created
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
            
            # Sort results by confidence (highest first)
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            return results
            
        except Exception as e:
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
            app.logger.error(f"Face recognition error: {str(e)}")
            return {"error": f"Face recognition failed: {str(e)}"}
            
    except Exception as e:
        app.logger.error(f"Unexpected error in recognize_faces_in_image: {str(e)}")
        return {"error": f"Image processing error: {str(e)}"}

def get_old_images():
    """Get list of old images for reminiscence therapy"""
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'gif']:
        image_files.extend(glob.glob(os.path.join(OLD_IMAGES_FOLDER, f"*.{ext}")))
    
    # Sort by modification time (newest first)
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    # Format for display in the template
    return [os.path.relpath(img, 'static').replace('\\', '/') for img in image_files]

def get_memory_prompts(num_prompts=3):
    """Generate memory prompts based on old photos"""
    prompts = [
        "What was happening in this photo?",
        "Where was this photo taken?",
        "Who were you with in this memory?",
        "What year do you think this was?",
        "What feelings does this photo bring up?",
        "What sounds or smells do you associate with this memory?",
        "What happened just before or after this photo was taken?",
        "What was life like during this period?",
        "What's your favorite part of this memory?",
        "How has this place changed since this photo was taken?",
        "What would you tell your younger self in this photo?"
    ]
    
    # Return random selection of prompts
    return random.sample(prompts, min(num_prompts, len(prompts)))

# Face Recognition Routes
@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition_page():
    """
    Renders the face recognition page and handles photo uploads for face detection
    """
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Initialize variables
    photo_path = None
    recognition_results = None
    
    # Check if reference faces exist
    reference_faces = get_reference_faces()
    reference_faces_exist = len(reference_faces) > 0
    
    if request.method == 'POST':
        # Process photo upload
        if 'photo' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)
        
        file = request.files['photo']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            try:
                # Generate a unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                
                # Create uploads directory if it doesn't exist
                uploads_dir = os.path.join(app.static_folder, 'uploads', 'faces')
                os.makedirs(uploads_dir, exist_ok=True)
                
                # Save the uploaded file
                file_path = os.path.join(uploads_dir, unique_filename)
                file.save(file_path)
                
                # Store relative path for display
                photo_path = f"uploads/faces/{unique_filename}"
                
                # Process the image for face recognition
                app.logger.info(f"Processing uploaded image for face recognition: {file_path}")
                recognition_results = recognize_faces_in_image(file_path)
                
                # Check if an error occurred
                if isinstance(recognition_results, dict) and 'error' in recognition_results:
                    error_msg = recognition_results['error']
                    app.logger.error(f"Face recognition error: {error_msg}")
                    flash(error_msg, 'danger')
                    recognition_results = None
                
                # Handle case where no faces were found
                elif not recognition_results:
                    flash('No faces were detected in the uploaded image.', 'warning')
            
            except Exception as e:
                app.logger.error(f"Error processing uploaded image: {str(e)}")
                flash(f'Error processing the image: {str(e)}', 'danger')
        else:
            flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
    
    # Render the face recognition page
    return render_template(
        'face_recognition.html',
        photo_path=photo_path,
        recognition_results=recognition_results,
        reference_faces_exist=reference_faces_exist
    )

@app.route('/add_reference_face', methods=['GET', 'POST'])
def add_reference_face():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    error_message = None
    success_message = None
    reference_faces = get_reference_faces()
    
    # Get source image from query parameter if provided
    source_image = request.args.get('source_image')
    if source_image and os.path.exists(os.path.join('static', source_image)):
        # Valid source image provided
        source_image_url = source_image
    else:
        source_image_url = None
    
    if request.method == 'POST':
        # Check if the post has the file part
        if 'photo' not in request.files:
            error_message = "No file part"
        else:
            file = request.files['photo']
            name = request.form.get('name', '').strip()
            relation = request.form.get('relation', '').strip()
            
            # Validate inputs
            if file.filename == '':
                error_message = "No selected file"
            elif not name:
                error_message = "Please provide a name for this person"
            else:
                try:
                    # Create unique filename using name and timestamp
                    safe_name = secure_filename(name.lower().replace(' ', '_'))
                    filename = f"{safe_name}_{int(time.time())}.jpg"
                    
                    # Save to uploads/references directory
                    filepath = os.path.join(REFERENCES_FOLDER, filename)
                    
                    # Save the uploaded file
                    file.save(filepath)
                    
                    # Load the image to verify it contains a face
                    image = face_recognition.load_image_file(filepath)
                    face_locations = face_recognition.face_locations(image)
                    
                    if not face_locations:
                        os.remove(filepath)
                        error_message = "No face detected in the uploaded image. Please try with a clearer photo."
                    else:
                        # Get face encoding
                        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                        
                        # Generate unique ID
                        face_id = str(uuid.uuid4())
                        
                        # Add to reference faces
                        new_face = {
                            "id": face_id,
                            "name": name,
                            "relation": relation,
                            "path": filepath,
                            "encoding": face_encoding
                        }
                        
                        # Update our database
                        reference_faces.append(new_face)
                        
                        # Convert to JSON serializable format
                        json_faces = []
                        for face in reference_faces:
                            json_face = face.copy()
                            if 'encoding' in json_face:
                                json_face['encoding'] = json_face['encoding'].tolist()
                            json_faces.append(json_face)
                        
                        # Save to database file
                        db_path = os.path.join(REFERENCES_FOLDER, 'faces.json')
                        with open(db_path, 'w') as f:
                            json.dump(json_faces, f)
                        
                        success_message = f"Successfully added {name} to the reference faces"
                        
                        # Refresh reference faces list
                        reference_faces = get_reference_faces()
                
                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"
    
    # Prepare data for template
    faces_data = []
    for i, face in enumerate(reference_faces):
        # Get relative path for display
        if 'path' in face:
            rel_path = os.path.relpath(face['path'], 'static').replace('\\', '/')
        else:
            rel_path = ''
        
        # Ensure each face has an ID (use index if not present)
        face_id = face.get('id', str(i))
        
        faces_data.append({
            'id': face_id,
            'name': face.get('name', 'Unknown'),
            'relation': face.get('relation', ''),
            'image_path': rel_path
        })
    
    context = {
        'error_message': error_message,
        'success_message': success_message,
        'reference_faces': faces_data,
        'source_image': source_image_url
    }
    
    return render_template('add_reference_face.html', **context)

@app.route('/delete_reference_face/<face_id>', methods=['POST'])
def delete_reference_face(face_id):
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        db_path = os.path.join(REFERENCES_FOLDER, 'faces.json')
        
        # Check if file exists
        if not os.path.exists(db_path):
            return jsonify({"error": "Face database not found"}), 404
        
        # Load existing faces
        with open(db_path, 'r') as f:
            try:
                reference_faces = json.load(f)
            except json.JSONDecodeError:
                return jsonify({"error": "Error reading face database"}), 500
        
        # Find the face with the matching ID
        found = False
        for i, face in enumerate(reference_faces):
            if str(face.get('id', '')) == str(face_id) or str(i) == str(face_id):
                # Delete the image file if it exists
                if 'path' in face and os.path.exists(face['path']):
                    try:
                        os.remove(face['path'])
                        print(f"Deleted file: {face['path']}")
                    except Exception as e:
                        print(f"Could not delete file: {e}")
                
                # Remove from list
                del reference_faces[i]
                found = True
                
                # Save updated list
                with open(db_path, 'w') as f:
                    json.dump(reference_faces, f)
                
                return jsonify({"success": True})
        
        if not found:
            return jsonify({"error": f"Face not found with ID: {face_id}"}), 404
        
    except Exception as e:
        print(f"Error deleting reference face: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Unknown error occurred"}), 500

# Reminiscence Therapy Routes
@app.route('/reminiscence_therapy')
def reminiscence_therapy():
    """Main page for reminiscence therapy"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get old images for the slideshow
    old_images = get_old_images()
    
    # Generate memory prompts
    memory_prompts = get_memory_prompts(3)
    
    # Get user info
    user_id = ObjectId(session['user_id'])
    user = get_user_data(user_id)
    
    # Get memory entries from database
    memory_entries = list(db.memory_entries.find({"user_id": user_id}).sort("date", -1))
    
    return render_template('reminiscence_therapy.html',
                           old_images=old_images,
                           memory_prompts=memory_prompts,
                           memory_entries=memory_entries,
                           user=user)

@app.route('/upload_memory', methods=['POST'])
def upload_memory():
    """Upload a memory (photo) for reminiscence therapy"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    
    if 'photo' not in request.files:
        flash("No file part", "danger")
        return redirect(url_for('reminiscence_therapy'))
    
    file = request.files['photo']
    year = request.form.get('year', '')
    description = request.form.get('description', '')
    
    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for('reminiscence_therapy'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            # Add year to filename if provided
            if year:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{year}{ext}"
            
            file_path = os.path.join(OLD_IMAGES_FOLDER, filename)
            file.save(file_path)
            
            # Resize the image if it's too large
            resize_image_if_needed(file_path, max_size=(1600, 1200))
            
            # Save metadata to database
            db.memory_entries.insert_one({
                "user_id": user_id,
                "filename": filename,
                "path": os.path.relpath(file_path, 'static').replace('\\', '/'),
                "year": year,
                "description": description,
                "date": datetime.now()
            })
            
            flash("Memory uploaded successfully!", "success")
        except Exception as e:
            flash(f"Error uploading memory: {str(e)}", "danger")
    else:
        flash("File type not allowed. Please upload a jpg, jpeg, png, or gif file.", "danger")
    
    return redirect(url_for('reminiscence_therapy'))

@app.route('/add_memory_entry', methods=['POST'])
def add_memory_entry():
    """Add a text memory entry without a photo"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    
    title = request.form.get('title', '')
    year = request.form.get('year', '')
    content = request.form.get('content', '')
    
    if not title or not content:
        flash("Title and content are required", "danger")
        return redirect(url_for('reminiscence_therapy'))
    
    try:
        # Save entry to database
        db.memory_entries.insert_one({
            "user_id": user_id,
            "title": title,
            "year": year,
            "content": content,
            "date": datetime.now()
        })
        
        flash("Memory entry added successfully!", "success")
    except Exception as e:
        flash(f"Error adding memory entry: {str(e)}", "danger")
    
    return redirect(url_for('reminiscence_therapy'))

@app.route('/api/voice_identify', methods=['POST'])
def voice_identify():
    """
    API endpoint to process voice queries about images for face recognition.
    Handles various error conditions with graceful fallbacks.
    """
    # Authentication check
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required', 'status_code': 401}), 401
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON data in request', 'status_code': 400}), 400
        
        image_path = data.get('image_path')
        query = data.get('query', '').lower()
        
        if not image_path:
            return jsonify({'error': 'Missing image path', 'status_code': 400}), 400
        
        # Construct full path
        full_path = os.path.join(app.static_folder, image_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            return jsonify({
                'response_text': "I'm sorry, I can't find the image. It may have been deleted or moved.",
                'error': 'Image file not found',
                'status_code': 404
            }), 404
        
        # Check if this is likely a face recognition query
        face_keywords = ['who', 'person', 'people', 'recognize', 'identify', 'face', 'name']
        is_face_query = any(keyword in query for keyword in face_keywords)
        
        if not is_face_query:
            return jsonify({
                'response_text': "I can answer questions about people in the photo. Try asking 'Who is this?' or 'Can you identify this person?'",
                'status_code': 200
            })
        
        # Process the image for face recognition
        app.logger.info(f"Processing image for face recognition: {image_path}")
        
        # Implement retries for face recognition
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Process image with face recognition
                recognition_results = recognize_faces_in_image(full_path)
                
                if isinstance(recognition_results, dict) and recognition_results.get('error'):
                    # Handle specific error from recognition function
                    error_msg = recognition_results.get('error')
                    app.logger.error(f"Face recognition error: {error_msg}")
                    
                    if 'no faces found' in error_msg.lower():
                        return jsonify({
                            'response_text': "I couldn't detect any faces in this image. The image might not contain people or the faces might not be clear enough.",
                            'status_code': 200
                        })
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            app.logger.info(f"Retrying face recognition (attempt {retry_count+1}/{max_retries})...")
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            return jsonify({
                                'response_text': f"I'm having trouble processing this image: {error_msg}",
                                'error': error_msg,
                                'status_code': 500
                            }), 500
                
                # If we reach here, recognition was successful
                break
                
            except Exception as e:
                app.logger.error(f"Error during face recognition: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    app.logger.info(f"Retrying face recognition (attempt {retry_count+1}/{max_retries})...")
                    time.sleep(1)  # Wait before retry
                else:
                    return jsonify({
                        'response_text': "I'm having trouble analyzing this image. There might be a technical issue.",
                        'error': str(e),
                        'status_code': 500
                    }), 500
        
        # Format response based on recognition results
        if not recognition_results or len(recognition_results) == 0:
            return jsonify({
                'response_text': "I don't recognize anyone in this photo. Try adding more reference faces or upload a clearer image.",
                'faces_found': 0,
                'status_code': 200
            })
        
        # Construct a natural language response
        if len(recognition_results) == 1:
            person = recognition_results[0]
            response_text = f"This appears to be {person['name']}"
            if person.get('relationship'):
                response_text += f", who is your {person['relationship']}"
            response_text += f". (Confidence: {person['confidence']*100:.1f}%)"
        else:
            response_text = f"I can see {len(recognition_results)} people in this photo: "
            for i, person in enumerate(recognition_results):
                if i > 0:
                    if i == len(recognition_results) - 1:
                        response_text += " and "
                    else:
                        response_text += ", "
                
                response_text += f"{person['name']}"
                if person.get('relationship'):
                    response_text += f" (your {person['relationship']})"
        
        return jsonify({
            'response_text': response_text,
            'faces_found': len(recognition_results),
            'recognition_results': recognition_results,
            'status_code': 200
        })
        
    except FileNotFoundError:
        return jsonify({
            'response_text': "I can't access the image file. It may have been moved or deleted.",
            'error': 'File not found',
            'status_code': 404
        }), 404
    except PermissionError:
        return jsonify({
            'response_text': "I don't have permission to access this file.",
            'error': 'Permission denied',
            'status_code': 403
        }), 403
    except json.JSONDecodeError:
        return jsonify({
            'response_text': "There was a problem understanding the request format.",
            'error': 'Invalid JSON',
            'status_code': 400
        }), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in voice_identify: {str(e)}")
        return jsonify({
            'response_text': "I encountered an unexpected error while processing your request.",
            'error': str(e),
            'status_code': 500
        }), 500

@app.route('/api/push-subscription', methods=['POST'])
def push_subscription():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        subscription = request.get_json()
        user_id = ObjectId(session['user_id'])
        
        # Update user's push subscription
        db.users.update_one(
            {"_id": user_id},
            {"$set": {"push_subscription": subscription}}
        )
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_voice_command', methods=['POST'])
def process_voice_command():
    """
    Process voice commands using Groq API to interpret natural language
    and return appropriate redirect URL
    """
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    user_id = session.get('user_id')
    user_type = session.get('user_type')
    data = request.get_json()
    command = data.get('command', '').strip()
    
    if not command:
        return jsonify({"status": "error", "message": "Empty command"}), 400
    
    try:
        # Initialize Groq client
        from groq import Groq
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        
        # Build available routes dictionary
        available_routes = {
            # Common routes for all users
            'dashboard': 'Main dashboard/home page',
            'logout': 'Sign out of the application',
            'tasks': 'Manage daily tasks and to-do items',
            'medications': 'Manage medication schedule',
            'memory_training': 'Access memory exercises and training',
            'notes': 'Access personal notes and reminders',
            'ai_assistant': 'Talk with the AI virtual assistant',
            'face_recognition': 'Identify people in photos',
            'reminiscence_therapy': 'View memories and old photos',
        }
        
        # Add patient-specific routes for caretakers
        patient_routes = {}
        if user_type == 'caretaker':
            patients = get_caretaker_patients(ObjectId(user_id))
            for patient in patients:
                patient_name = patient['name']
                patient_id = str(patient['_id'])
                patient_routes[f"manage_patient/{patient_id}"] = f"Manage patient {patient_name}"
        
        # Construct the prompt for Groq
        route_descriptions = "\n".join([f"- {desc} → {route}" for route, desc in available_routes.items()])
        patient_descriptions = ""
        if patient_routes:
            patient_descriptions = "\n".join([f"- {desc} → {route}" for route, desc in patient_routes.items()])
            
        prompt = f"""You are an assistant for a healthcare application called MemoryCare.
        
Given a voice command, determine which page/route the user wants to navigate to.

Available routes:
{route_descriptions}

{"Patient-specific routes:" if patient_routes else ""}
{patient_descriptions}

For the voice command: "{command}"

Return only the route name (e.g., "dashboard", "tasks", etc.) without any explanation or additional text. If uncertain, return "unknown".
"""

        # Call Groq API
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=50,
            top_p=1,
            stream=False,
        )
        
        # Extract the route from the response
        route = completion.choices[0].message.content.strip()
        
        # Handle the response
        if route in available_routes:
            return jsonify({
                "status": "success",
                "command": command,
                "redirect": url_for(route),
                "message": f"Navigating to {available_routes[route]}"
            })
        elif route in patient_routes:
            # For patient routes, we need to extract the patient_id
            patient_path = route.split('/')
            if len(patient_path) == 2:
                return jsonify({
                    "status": "success",
                    "command": command,
                    "redirect": url_for('manage_patient', patient_id=patient_path[1]),
                    "message": f"Navigating to {patient_routes[route]}"
                })
        elif route == "unknown":
            return jsonify({
                "status": "error",
                "command": command,
                "message": "I'm not sure where you want to go. Please try again with a clearer command."
            })
        else:
            # Try to see if it's a partial match to any route
            for available_route in list(available_routes.keys()) + list(patient_routes.keys()):
                if route.lower() in available_route.lower():
                    if "manage_patient" in available_route:
                        patient_path = available_route.split('/')
                        return jsonify({
                            "status": "success",
                            "command": command,
                            "redirect": url_for('manage_patient', patient_id=patient_path[1]),
                            "message": f"Navigating to {patient_routes.get(available_route, 'patient management')}"
                        })
                    else:
                        return jsonify({
                            "status": "success",
                            "command": command,
                            "redirect": url_for(available_route),
                            "message": f"Navigating to {available_routes.get(available_route, available_route)}"
                        })
            
            return jsonify({
                "status": "error",
                "command": command,
                "message": f"Command not recognized: '{command}'. Please try again."
            })
            
    except Exception as e:
        print(f"Error processing voice command: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": "An error occurred while processing your request. Please try again.",
            "details": str(e)
        })

@app.route('/api/upload_photo', methods=['POST'])
def api_upload_photo():
    """API endpoint for uploading photos with progress tracking"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        if 'photo' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['photo']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            # Save the uploaded file with a unique name
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file.save(file_path)
            
            # Resize the image if it's too large
            resize_image_if_needed(file_path)
            
            # Return the file path for further processing
            return jsonify({
                "status": "success",
                "filename": unique_filename,
                "path": os.path.relpath(file_path, 'static').replace('\\', '/')
            })
        else:
            return jsonify({"error": "File type not allowed"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    try:
        if 'image_path' in request.json:
            # Process existing image
            image_path = request.json['image_path']
            full_path = os.path.join('static', image_path)
            
            if not os.path.exists(full_path):
                return jsonify({"error": f"Image file not found: {image_path}"}), 404
                
            return jsonify(recognize_faces_in_image(full_path))
        else:
            # Direct processing from uploaded file
            if 'photo' not in request.files:
                return jsonify({"error": "No file part"}), 400
                
            file = request.files['photo']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
                
            # Check file type
            if not allowed_file(file.filename):
                return jsonify({"error": "File type not allowed. Please use jpg, jpeg, or png."}), 400
                
            # Check file size
            if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({"error": f"File is too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)}MB"}), 400
                
            # Create a unique filename
            filename = str(uuid.uuid4()) + secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save file
            file.save(filepath)
            
            # Process the image
            result = recognize_faces_in_image(filepath)
            
            # Include the path in the response
            result['image_path'] = os.path.relpath(filepath, 'static').replace('\\', '/')
            return jsonify(result)
            
    except Exception as e:
        print(f"Error in recognize_face route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

def get_reference_faces():
    """Get all reference faces from the local filesystem"""
    try:
        # Initialize references folder and database
        reference_dir = REFERENCES_FOLDER
        os.makedirs(reference_dir, exist_ok=True)
        db_path = os.path.join(reference_dir, 'faces.json')
        
        # If the database exists, load it
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                try:
                    reference_faces = json.load(f)
                    # Convert string encoding back to numpy arrays
                    for i, face in enumerate(reference_faces):
                        # Ensure each face has an id
                        if 'id' not in face:
                            face['id'] = str(i)
                        if 'encoding' in face and isinstance(face['encoding'], list):
                            face['encoding'] = np.array(face['encoding'])
                    return reference_faces
                except json.JSONDecodeError:
                    print("Error decoding reference faces JSON. Creating a new database.")
                    # Create empty database
                    with open(db_path, 'w') as f:
                        json.dump([], f)
                    return []
        else:
            # Create empty database file
            with open(db_path, 'w') as f:
                json.dump([], f)
            print("Created new reference face database")
            
            # Scan for existing images in the references directory
            reference_faces = []
            
            # Find all image files in references directory
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(reference_dir, f'*{ext}')))
                
            # If we found images, process them
            if image_files:
                print(f"Found {len(image_files)} images in references directory, processing...")
                
                for i, image_path in enumerate(image_files):
                    try:
                        # Extract name from filename (remove extension)
                        name = os.path.splitext(os.path.basename(image_path))[0]
                        
                        # Load image and find face
                        image = face_recognition.load_image_file(image_path)
                        
                        # Find all faces in the image
                        face_locations = face_recognition.face_locations(image)
                        
                        if not face_locations:
                            print(f"No face found in {image_path}")
                            continue
                        
                        # Use the first face found (assuming one face per reference image)
                        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                        
                        # Add to reference faces with unique ID
                        face_id = str(uuid.uuid4())
                        reference_faces.append({
                            "id": face_id,
                            "name": name,
                            "relation": "",  # No relation info from filename
                            "path": image_path,
                            "encoding": face_encoding
                        })
                        
                        print(f"Added reference face for {name}")
                        
                    except Exception as e:
                        print(f"Error processing reference image {image_path}: {str(e)}")
                
                # Save the database with the processed images
                if reference_faces:
                    # Convert numpy arrays to lists for JSON serialization
                    json_faces = []
                    for face in reference_faces:
                        json_face = face.copy()
                        if 'encoding' in json_face:
                            json_face['encoding'] = json_face['encoding'].tolist()
                        json_faces.append(json_face)
                        
                    with open(db_path, 'w') as f:
                        json.dump(json_faces, f)
                    
                    # Convert back to numpy arrays for return
                    for face in reference_faces:
                        if 'encoding' in face and isinstance(face['encoding'], list):
                            face['encoding'] = np.array(face['encoding'])
            
            return reference_faces
        
    except Exception as e:
        print(f"Error in get_reference_faces: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Add a backward-compatible route for face_recognition
@app.route('/face_recognition', methods=['GET', 'POST'])
def face_recognition_redirect():
    """Redirect from old route to new hyphenated route"""
    return redirect(url_for('face_recognition_page'))

# Add a route alias for user_dashboard
@app.route('/user_dashboard')
def user_dashboard():
    """Alias for the dashboard route"""
    return redirect(url_for('dashboard'))

# Debug route to list all registered routes
@app.route('/debug/routes')
def debug_routes():
    """List all registered routes for debugging"""
    if app.debug:
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': ','.join(rule.methods),
                'path': str(rule)
            })
        return jsonify(routes)
    return "Debug mode is not enabled", 403

# Authentication helper
def requires_auth(f):
    """Decorator to check if user is authenticated"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)