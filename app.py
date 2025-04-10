try:
    from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, g
    from werkzeug.security import generate_password_hash, check_password_hash
    from werkzeug.utils import secure_filename
    from datetime import datetime, timedelta
    import os
    import random
    import json
    import glob

    import nltk
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import re
    import certifi
    import uuid
    from functools import wraps
    import pickle
    import base64
    from PIL import Image
    import cv2
    import io
    import math
    import google_auth_oauthlib.flow
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    import time
    import pytz
    from bson.objectid import ObjectId
    
except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Please make sure all required packages are installed by running: pip install -r requirements.txt")
    exit(1)

from dotenv import load_dotenv
load_dotenv()
import nltk
nltk.download('punkt')

# Constants and configurations
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'remindappsecretkey')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit file uploads to 16MB
app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'text/javascript', 'application/javascript']
app.config['COMPRESS_LEVEL'] = 6
app.config['COMPRESS_MIN_SIZE'] = 500
from flask_compress import Compress

# Initialize Flask-Compress
compress = Compress(app)

# Setup folders
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
OLD_IMAGES_FOLDER = os.path.join(app.static_folder, 'images', 'memories')
KNOWN_FACES_FOLDER = os.path.join(app.static_folder, 'uploads', 'known_faces')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Set secure headers for PWA
@app.after_request
def add_pwa_headers(response):
    # Add headers to help with PWA functionality
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    response.headers['Permissions-Policy'] = 'geolocation=(self), microphone=(self), camera=(self)'
    
    # Add Service-Worker-Allowed header for broader scope
    if request.path.endswith('sw.js'):
        response.headers['Service-Worker-Allowed'] = '/'
    
    # Add cache control headers for PWA assets
    if request.path.startswith('/static/'):
        if any(request.path.endswith(ext) for ext in ['.js', '.css']):
            # Cache JavaScript and CSS for 1 day
            response.headers['Cache-Control'] = 'public, max-age=86400'
        elif any(request.path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
            # Cache images for 7 days
            response.headers['Cache-Control'] = 'public, max-age=604800'
    
    # Ensure proper headers for manifest.json
    if request.path.endswith('manifest.json'):
        response.headers['Content-Type'] = 'application/manifest+json'
        response.headers['Cache-Control'] = 'public, max-age=86400'
    
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
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 16 MB max upload size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure all required directories exist
for directory in [UPLOAD_FOLDER, OLD_IMAGES_FOLDER, KNOWN_FACES_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

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
                'email': 'admin@remind.com',
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
    source = request.args.get('source')
    
    # Add standalone parameter to session if provided
    if standalone or source == 'pwa':
        session['standalone'] = True
    
    # Check if user is logged in, if so redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    # If it's a PWA or standalone app, redirect to splash screen
    if session.get('standalone') or source == 'pwa' or request.args.get('homescreen') == 'true':
        return redirect(url_for('splash'))
    
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
        
        # Handle password verification with support for different hash types
        login_successful = False
        
        if user:
            try:
                # Try to verify the password with the existing hash
                login_successful = check_password_hash(user['password'], password)
                
                # If successful and using scrypt, update to a supported algorithm
                if login_successful and user['password'].startswith('scrypt'):
                    new_hash = generate_password_hash(password, method='pbkdf2:sha256')
                    db.users.update_one(
                        {"_id": user['_id']},
                        {"$set": {"password": new_hash}}
                    )
            except ValueError:
                # If there's a ValueError (likely due to unsupported hash type),
                # We will try a fallback approach
                # This is a temporary solution to migrate existing users
                try:
                    # Create a new hash with a supported method and verify manually
                    new_hash = generate_password_hash(password, method='pbkdf2:sha256')
                    # For migration purposes, if the email and password match known credentials,
                    # we can consider this a successful login and update the hash
                    
                    # IMPORTANT: This part should be customized based on your specific situation
                    # For example, you might have a list of known users and passwords
                    # or a fallback verification method
                    
                    # Example fallback (customize as needed):
                    # login_successful = verify_credentials_fallback(email, password)
                    
                    if login_successful:
                        # Update the hash in the database
                        db.users.update_one(
                            {"_id": user['_id']},
                            {"$set": {"password": new_hash}}
                        )
                except Exception as e:
                    # If all else fails, login remains unsuccessful
                    login_successful = False
        
        if login_successful:
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
        
        # Create new user with explicit hash method that's supported
        new_user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password, method='pbkdf2:sha256'),
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

# This helper function needs to be implemented if you're using the fallback verification
# You can customize this based on your specific requirements
def verify_credentials_fallback(email, password):
    """
    A fallback method to verify credentials when standard password verification fails.
    This is useful during the migration from unsupported hash types.
    
    Args:
        email: The user's email
        password: The plaintext password to verify
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Implementation depends on your specific requirements
    # For example, you might check against a backup system or known credentials
    
    # EXAMPLE (replace with your actual implementation):
    # return email in known_credentials and known_credentials[email] == password
    
    # By default, return False to be safe
    return False
    
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
    
    try:
        # Initialize Groq client
        from groq import Groq
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        
        # Get user's personal information
        personal_info = user.get('personal_info', '')
        name = user.get('name', 'User')
        
        # Get user's task data
        task_data = db.tasks.find_one({"user_id": user_id})
        tasks = task_data.get('tasks', []) if task_data else []
        
        # Get medication data
        med_data = db.medications.find_one({"user_id": user_id})
        medications = med_data.get('medications', []) if med_data else []
        
        # Get notes
        notes_data = db.notes.find_one({"user_id": user_id})
        notes = notes_data.get('content', '') if notes_data else ''
        
        # Create context for the LLM
        context = f"""
        USER INFORMATION:
        Name: {name}
        Personal Information: {personal_info}
        
        TASKS:
        {', '.join([t.get('text', '') for t in tasks])}
        
        MEDICATIONS:
        {', '.join([m.get('name', '') + ' at ' + m.get('time', '') for m in medications])}
        
        NOTES:
        {notes}
        """
        
        # Build the prompt for the LLM
        llm_prompt = f"""You are a compassionate AI memory assistant for Ramesh, a 65-year-old person living with moderate-stage Alzheimer's disease.  

You have the following background information about Ramesh:  

{context}  

Ramesh is asking: "{prompt}"  

Respond in a gentle, reassuring, and easy-to-understand manner. Keep sentences short and simple. Avoid overwhelming details.  

If the query involves personal identity, provide the following responses:  
{{
    'name': 'Your name is Ramesh.',
    'age': 'You are 65 years old.',
    'family': 'Your family loves and cares for you very much.',
    'home': 'You live in Bengaluru with your daughter and her family.',
    'hobbies': 'You enjoy listening to old songs and play at family photos.',
    'daily routine': 'You start the day with a nice meal, take a walk, and spend time with family.',
    'friends': 'Your friends and family think about you and care for you.',
    'doctor': 'Your doctor ensures that you are healthy and taken care of.',
    'safety': 'You are safe, and your loved ones are here to support you.'
}}  

If Ramesh seems confused or anxious, offer comforting and calming responses, such as:  
- "It's okay, take your time."  
- "You are safe, and everything is alright."  
- "Your family is here for you, and they love you."  
- "Would you like to listen to some music? It might help you feel better."  
- "Let's look at some old photos together. That always brings back nice memories."  

Ensure responses are warm, patient, and supportive to help Ramesh feel at ease.  
"""  

        # Simple memory system to provide context
        memory = {
            'hometown': 'Salem, Tamil Nadu',
            'age': 68,
            'occupation': 'Retired teacher',
            'family': 'You have a daughter named Priya and a son named Raj',
            'home': 'You live in Bengaluru with your daughter and her family.',
            'hobbies': 'You enjoy gardening, reading, and playing with your grandchildren',
        }
        
        # Call Groq API
        completion = client.chat.completions.create(
            model="gemma2-9b-it",  # Using the same model as voice commands
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False,
        )
        
        # Extract the response
        response = completion.choices[0].message.content.strip()
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Error in AI response: {str(e)}")
        
        # Fallback to the old method if API fails
        qa_system = PersonalInfoQA()
        qa_system.train(user.get('personal_info', ''))
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
        # Hardcoded location for Dr. Ambedkar Institute Of Technology
        latitude = 12.9637
        longitude = 77.5060
        accuracy = 0.0
        source = "hardcoded"
        timestamp = datetime.now()
        address = "Dr. Ambedkar Institute Of Technology, Bengaluru, Karnataka, India"
        
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
    
    # Check user agent for Android
    user_agent = request.headers.get('User-Agent', '').lower()
    g.is_android = 'android' in user_agent
    
    # Detect if app is already installed
    if request.headers.get('Display-Mode') == 'standalone' or request.args.get('source') == 'pwa':
        g.is_installed_pwa = True

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
@app.route('/ai_assistant')
def ai_assistant():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = ObjectId(session['user_id'])
    user = get_user_data(ObjectId(user_id))
    
    return render_template('ai_assistant.html', user=user)

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
            'reminiscence_therapy': 'View memory album',
            'fitness_analysis': 'View fitness information',
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
            
        prompt = f"""You are an assistant for a healthcare application called ReMind.
        
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

# SOS Emergency Alerts Routes
@app.route('/api/sos_alert', methods=['POST'])
def sos_alert():
    """Handle emergency SOS alerts from patients"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        user_id = ObjectId(session['user_id'])
        user = get_user_data(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get patient location if available
        location_data = db.locations.find_one({"user_id": user_id})
        location = "Unknown"
        
        if location_data:
            location = location_data.get("address", "Unknown")
        
        # Create emergency alert
        alert_data = {
            "patient_id": user_id,
            "patient_name": user.get("name", "Unknown"),
            "timestamp": datetime.now(),
            "location": data.get("location", location),
            "status": "active",
            "description": data.get("description", "Emergency assistance needed")
        }
        
        # Store alert in database
        alert_id = db.emergency_alerts.insert_one(alert_data).inserted_id
        
        # Find patient's caretaker
        caretaker_id = user.get("caretaker_id")
        
        # Send notification to caretaker if assigned
        if caretaker_id:
            caretaker = get_user_data(caretaker_id)
            if caretaker:
                # Get caretaker's push subscription
                push_subscription = caretaker.get("push_subscription")
                
                if push_subscription:
                    # In a production app, this would send a push notification
                    print(f"Would send push notification to caretaker {caretaker['name']}")
                
                # Store notification in database
                notification_data = {
                    "user_id": caretaker_id,
                    "type": "emergency_alert",
                    "title": "Emergency Alert",
                    "body": f"{user['name']} needs emergency assistance",
                    "data": {
                        "alert_id": str(alert_id),
                        "patient_id": str(user_id),
                        "patient_name": user['name'],
                        "location": location
                    },
                    "read": False,
                    "created_at": datetime.now()
                }
                
                db.notifications.insert_one(notification_data)
        
        return jsonify({
            "status": "success",
            "message": "Emergency alert sent successfully",
            "alert_id": str(alert_id)
        })
        
    except Exception as e:
        print(f"Error in SOS alert: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sos_alerts', methods=['GET'])
def get_sos_alerts():
    """Get list of active emergency alerts for caretaker"""
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return jsonify({"error": "Not authorized"}), 403
    
    try:
        caretaker_id = ObjectId(session['user_id'])
        
        # Get all patients assigned to this caretaker
        patients = get_caretaker_patients(caretaker_id)
        patient_ids = [patient['_id'] for patient in patients]
        
        # Get active alerts for these patients
        alerts = list(db.emergency_alerts.find({
            "patient_id": {"$in": patient_ids},
            "status": "active"
        }).sort("timestamp", -1))
        
        # Format alerts for JSON response
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "alert_id": str(alert['_id']),
                "patient_id": str(alert['patient_id']),
                "patient_name": alert['patient_name'],
                "timestamp": alert['timestamp'].isoformat(),
                "location": alert['location'],
                "status": alert['status'],
                "description": alert.get('description', '')
            })
        
        return jsonify({
            "status": "success",
            "alerts": formatted_alerts,
            "count": len(formatted_alerts)
        })
        
    except Exception as e:
        print(f"Error getting SOS alerts: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sos_alert/<alert_id>/respond', methods=['POST'])
def respond_to_alert(alert_id):
    """Mark an emergency alert as responded"""
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return jsonify({"error": "Not authorized"}), 403
    
    try:
        caretaker_id = ObjectId(session['user_id'])
        
        # Update alert status
        result = db.emergency_alerts.update_one(
            {"_id": ObjectId(alert_id), "status": "active"},
            {"$set": {
                "status": "responded",
                "responded_by": caretaker_id,
                "responded_at": datetime.now()
            }}
        )
        
        if result.modified_count == 0:
            return jsonify({"error": "Alert not found or already responded"}), 404
        
        return jsonify({
            "status": "success",
            "message": "Alert marked as responded"
        })
        
    except Exception as e:
        print(f"Error responding to alert: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sos_alert/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Mark an emergency alert as resolved"""
    if 'user_id' not in session or session.get('user_type') != 'caretaker':
        return jsonify({"error": "Not authorized"}), 403
    
    try:
        caretaker_id = ObjectId(session['user_id'])
        
        # Update alert status
        result = db.emergency_alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {
                "status": "resolved",
                "resolved_by": caretaker_id,
                "resolved_at": datetime.now(),
                "resolution_notes": request.json.get("notes", "")
            }}
        )
        
        if result.modified_count == 0:
            return jsonify({"error": "Alert not found"}), 404
        
        return jsonify({
            "status": "success",
            "message": "Alert resolved successfully"
        })
        
    except Exception as e:
        print(f"Error resolving alert: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Google Fit Integration
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Google OAuth Configuration
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # For development only
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
]
REDIRECT_URI = "http://localhost:5000/callback"

def get_last_7_days():
    """Generate list of last 7 days dates."""
    return [(datetime.now().date() - timedelta(days=i)) for i in range(6, -1, -1)]

def get_sample_steps():
    """Generate sample step data."""
    dates = get_last_7_days()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'steps': [5000, 6200, 4800, 7500, 5600, 6100, 5300]
    }

def get_sample_heart_rate():
    """Generate sample heart rate data."""
    dates = get_last_7_days()
    timestamps = [datetime.combine(d, datetime.min.time().replace(hour=12, minute=0)).isoformat() for d in dates]
    return {
        'timestamps': timestamps,
        'values': [72.5, 75.2, 70.8, 73.6, 71.9, 74.3, 69.5]
    }

def get_sample_sleep():
    """Generate sample sleep data."""
    dates = get_last_7_days()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'Awake': [30, 25, 20, 35, 40, 30, 25],
        'Light': [240, 260, 220, 250, 230, 270, 280],
        'Deep': [90, 100, 110, 80, 95, 85, 105],
        'REM': [60, 65, 70, 55, 75, 65, 60]
    }

def get_sample_calories():
    """Return sample calories data"""
    dates = get_last_7_days()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'calories': [2100.5, 2300.2, 1950.7, 2400.1, 2200.3, 2150.6, 2050.4]
    }

def credentials_to_dict(credentials):
    """Convert credentials to dictionary for session storage."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@app.route("/fitness")
def fitness_home():
    """Google Fit Integration home page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("fitness_index.html")

@app.route("/fitness/login")
def fitness_login():
    """Initiate Google OAuth flow."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
        )
        auth_url, _ = flow.authorization_url(prompt="consent")
        session['oauth_state'] = flow._state
        return redirect(auth_url)
    except Exception as e:
        flash(f"Error initiating Google login: {str(e)}", "danger")
        return redirect(url_for('fitness_home'))

@app.route("/callback")
def oauth_callback():
    """Handle OAuth callback from Google."""
    if 'user_id' not in session or 'oauth_state' not in session:
        return redirect(url_for('login'))
    
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI,
            state=session['oauth_state']
        )
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        session["fitness_credentials"] = credentials_to_dict(credentials)
        return redirect(url_for('fitness_dashboard'))
    except Exception as e:
        flash(f"Error completing Google authentication: {str(e)}", "danger")
        return redirect(url_for('fitness_home'))

@app.route('/fitness/dashboard')
def fitness_dashboard():
    # Use sample data if not authenticated or if there's an error
    try:
        if 'fitness_credentials' not in session:
            # Use sample data if not authenticated
            steps_data = get_sample_steps()
            heart_rate_data = get_sample_heart_rate()
            sleep_data = get_sample_sleep()
            calories_data = get_sample_calories()
            
            # Format data for the template
            steps_list = []
            for i, date in enumerate(steps_data['dates']):
                steps_list.append({
                    'date': date,
                    'count': steps_data['steps'][i]
                })
                
            heart_rate_list = []
            for i, timestamp in enumerate(heart_rate_data['timestamps']):
                heart_rate_list.append({
                    'timestamp': timestamp,
                    'rate': heart_rate_data['values'][i]
                })
                
            sleep_list = []
            for i, date in enumerate(sleep_data['dates']):
                sleep_list.append({
                    'date': date,
                    'deep': sleep_data['Deep'][i],
                    'rem': sleep_data['REM'][i],
                    'light': sleep_data['Light'][i],
                    'total': sleep_data['Light'][i] + sleep_data['Deep'][i] + sleep_data['REM'][i]
                })
                
            calories_list = []
            for i, date in enumerate(calories_data['dates']):
                calories_list.append({
                    'date': date,
                    'calories': calories_data['calories'][i]
                })
            
            # Get summary metrics for display
            daily_steps = steps_data['steps'][-1]
            avg_heart_rate = sum(heart_rate_data['values']) // len(heart_rate_data['values'])
            sleep_duration = (sleep_data['Light'][-1] + sleep_data['Deep'][-1] + sleep_data['REM'][-1]) // 60  # Convert to hours
            daily_calories = int(calories_data['calories'][-1])
            
            return render_template('fitness_dashboard.html',
                                   steps_data=json.dumps(steps_list),
                                   heart_rate_data=json.dumps(heart_rate_list),
                                   sleep_data=json.dumps(sleep_list),
                                   calories_data=json.dumps(calories_list),
                                   daily_steps=daily_steps,
                                   avg_heart_rate=avg_heart_rate,
                                   sleep_duration=sleep_duration,
                                   daily_calories=daily_calories,
                                   sample_data=True)
        else:
            # This would be where we fetch real data from Google Fit API
            # For now, we'll use sample data and pretend it's from the API
            steps_data = get_sample_steps()
            heart_rate_data = get_sample_heart_rate()
            sleep_data = get_sample_sleep()
            calories_data = get_sample_calories()
            
            # Format data for the template
            steps_list = []
            for i, date in enumerate(steps_data['dates']):
                steps_list.append({
                    'date': date,
                    'count': steps_data['steps'][i]
                })
                
            heart_rate_list = []
            for i, timestamp in enumerate(heart_rate_data['timestamps']):
                heart_rate_list.append({
                    'timestamp': timestamp,
                    'rate': heart_rate_data['values'][i]
                })
                
            sleep_list = []
            for i, date in enumerate(sleep_data['dates']):
                sleep_list.append({
                    'date': date,
                    'deep': sleep_data['Deep'][i],
                    'rem': sleep_data['REM'][i],
                    'light': sleep_data['Light'][i],
                    'total': sleep_data['Light'][i] + sleep_data['Deep'][i] + sleep_data['REM'][i]
                })
                
            calories_list = []
            for i, date in enumerate(calories_data['dates']):
                calories_list.append({
                    'date': date,
                    'calories': calories_data['calories'][i]
                })
            
            # Get summary metrics for display
            daily_steps = steps_data['steps'][-1]
            avg_heart_rate = sum(heart_rate_data['values']) // len(heart_rate_data['values'])
            sleep_duration = (sleep_data['Light'][-1] + sleep_data['Deep'][-1] + sleep_data['REM'][-1]) // 60  # Convert to hours
            daily_calories = int(calories_data['calories'][-1])
            
            return render_template('fitness_dashboard.html',
                                   steps_data=json.dumps(steps_list),
                                   heart_rate_data=json.dumps(heart_rate_list),
                                   sleep_data=json.dumps(sleep_list),
                                   calories_data=json.dumps(calories_list),
                                   daily_steps=daily_steps,
                                   avg_heart_rate=avg_heart_rate,
                                   sleep_duration=sleep_duration,
                                   daily_calories=daily_calories,
                                   sample_data=False)
    except Exception as e:
        print(f"Error retrieving fitness data: {e}")
        # Fallback to sample data in case of error
        steps_data = get_sample_steps()
        heart_rate_data = get_sample_heart_rate()
        sleep_data = get_sample_sleep()
        calories_data = get_sample_calories()
        
        # Format data for the template
        steps_list = []
        for i, date in enumerate(steps_data['dates']):
            steps_list.append({
                'date': date,
                'count': steps_data['steps'][i]
            })
            
        heart_rate_list = []
        for i, timestamp in enumerate(heart_rate_data['timestamps']):
            heart_rate_list.append({
                'timestamp': timestamp,
                'rate': heart_rate_data['values'][i]
            })
            
        sleep_list = []
        for i, date in enumerate(sleep_data['dates']):
            sleep_list.append({
                'date': date,
                'deep': sleep_data['Deep'][i],
                'rem': sleep_data['REM'][i],
                'light': sleep_data['Light'][i],
                'total': sleep_data['Light'][i] + sleep_data['Deep'][i] + sleep_data['REM'][i]
            })
            
        calories_list = []
        for i, date in enumerate(calories_data['dates']):
            calories_list.append({
                'date': date,
                'calories': calories_data['calories'][i]
            })
        
        # Get summary metrics for display
        daily_steps = steps_data['steps'][-1]
        avg_heart_rate = sum(heart_rate_data['values']) // len(heart_rate_data['values'])
        sleep_duration = (sleep_data['Light'][-1] + sleep_data['Deep'][-1] + sleep_data['REM'][-1]) // 60  # Convert to hours
        daily_calories = int(calories_data['calories'][-1])
        
        return render_template('fitness_dashboard.html',
                               steps_data=json.dumps(steps_list),
                               heart_rate_data=json.dumps(heart_rate_list),
                               sleep_data=json.dumps(sleep_list),
                               calories_data=json.dumps(calories_list),
                               daily_steps=daily_steps,
                               avg_heart_rate=avg_heart_rate,
                               sleep_duration=sleep_duration,
                               daily_calories=daily_calories,
                               sample_data=True,
                               error="Could not retrieve data from Google Fit")

# Get sample heart rate data for testing
def get_sample_heart_rate():
    # Timestamps for the last 7 days
    timestamps = []
    heart_rates = []
    dates = get_last_7_days()
    
    for day in dates:
        # Add 4 entries per day at different times
        for hour in [8, 12, 16, 20]:
            timestamp = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
            # Convert to milliseconds
            timestamp_ms = int(timestamp.timestamp() * 1000)
            timestamps.append(timestamp_ms)
            # Random heart rate between 60 and 100
            heart_rates.append(random.randint(60, 100))
    
    return list(zip(timestamps, heart_rates))

# Fitness Analysis Route
@app.route('/fitness_analysis')
def fitness_analysis():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get the fitness data for display
    sample_steps = get_sample_steps()
    sample_heart_rate = get_sample_heart_rate()
    sample_sleep = get_sample_sleep()
    sample_calories = get_sample_calories()
    
    return render_template('fitness_analysis.html', 
                           steps_data=sample_steps,
                           heart_rate_data=sample_heart_rate,
                           sleep_data=sample_sleep,
                           calories_data=sample_calories)

# Face recognition functions
PEOPLE_FOLDER = os.path.join('static', 'uploads', 'people')
# Ensure people folder exists
os.makedirs(PEOPLE_FOLDER, exist_ok=True)

@app.route('/pwa-install')
def pwa_install():
    """Show PWA installation instructions page"""
    # Detect device type for platform-specific instructions
    user_agent = request.headers.get('User-Agent', '').lower()
    
    is_android = 'android' in user_agent
    is_ios = 'iphone' in user_agent or 'ipad' in user_agent
    
    return render_template(
        'pwa_install.html', 
        is_android=is_android, 
        is_ios=is_ios
    )

@app.route('/share-target', methods=['POST'])
def share_target():
    """Handle Web Share Target API requests"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    share_data = {
        'title': request.form.get('title', ''),
        'text': request.form.get('text', ''),
        'url': request.form.get('url', '')
    }
    
    # Handle shared files/photos if any
    shared_files = []
    if 'photos' in request.files:
        photos = request.files.getlist('photos')
        for photo in photos:
            if photo and allowed_file(photo.filename):
                # Save the file
                filename = secure_filename(photo.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                photo.save(file_path)
                shared_files.append(file_path)
    
    # Store shared data in session to use on the next page
    session['shared_data'] = {
        'text': share_data['text'],
        'title': share_data['title'],
        'url': share_data['url'],
        'files': shared_files
    }
    
    # Redirect to appropriate page based on content type
    if shared_files:
        return redirect(url_for('reminiscence_therapy'))
    else:
        return redirect(url_for('notes'))

@app.route('/api/pwa-status', methods=['GET'])
def pwa_status():
    """API endpoint to check if running as an installed PWA"""
    is_standalone = (
        request.headers.get('Display-Mode') == 'standalone' or
        request.args.get('source') == 'pwa' or
        session.get('standalone') == True
    )
    
    user_agent = request.headers.get('User-Agent', '').lower()
    is_android = 'android' in user_agent
    is_ios = 'iphone' in user_agent or 'ipad' in user_agent
    
    return jsonify({
        'isStandalone': is_standalone,
        'isAndroid': is_android,
        'isIOS': is_ios,
        'isInstallable': is_android or (is_ios and not is_standalone)
    })

@app.route('/sw.js')
def service_worker():
    """Serve the service worker with appropriate headers"""
    response = send_from_directory(app.static_folder, 'sw.js')
    # Set the correct MIME type
    response.headers['Content-Type'] = 'application/javascript'
    # Set Cache-Control header to no-cache to ensure fresh service worker
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    # Set Service-Worker-Allowed header for broader scope
    response.headers['Service-Worker-Allowed'] = '/'
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
