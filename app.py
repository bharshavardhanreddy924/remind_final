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

except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Please make sure all required packages are installed by running: pip install -r requirements.txt")
    exit(1)

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
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
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
            
            db.tasks.update_one(
                {"user_id": user_id},
                {"$push": {"tasks": {"text": task_text, "completed": False}}}
            )
            flash('Task added successfully', 'success')
        
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
                    
                    db.medications.update_one(
                        {"user_id": user_id},
                        {"$push": {"medications": new_med}},
                        upsert=True
                    )
                    
                    flash('Medication added successfully', 'success')
                except ValueError:
                    flash('Invalid time format. Please use HH:MM AM/PM', 'danger')
        
        elif request.form.get('action') == 'delete':
            med_id = request.form.get('med_id')
            
            db.medications.update_one(
                {"user_id": user_id},
                {"$pull": {"medications": {"id": med_id}}}
            )
            
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

if __name__ == '__main__':
    app.run(debug=True)
