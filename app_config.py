import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/remind_db')

# Server settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 8080))
HOST = os.getenv('HOST', '0.0.0.0')

# App settings
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Storage paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
KNOWN_FACES_FOLDER = os.path.join('static', 'uploads', 'known_faces')
MEMORIES_FOLDER = os.path.join('static', 'uploads', 'memories')

# File upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 