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
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)