import os
import datetime
import logging
from flask import Flask, redirect, request, session, render_template
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

app = Flask(__name__)
app.secret_key = "your_secret_key"

CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
]
REDIRECT_URI = "http://localhost:5000/callback"

flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
)

def get_last_7_days():
    """Generate list of last 7 days dates."""
    return [(datetime.datetime.utcnow().date() - datetime.timedelta(days=i)) for i in range(6, -1, -1)]

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
    timestamps = [datetime.datetime.combine(d, datetime.time(12, 0)).isoformat() for d in dates]
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
    """Generate sample calories data."""
    dates = get_last_7_days()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'calories': [2100.5, 2300.2, 1950.7, 2400.1, 2200.3, 2150.6, 2050.4]
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    auth_url, _ = flow.authorization_url(prompt="consent")
    return redirect(auth_url)

@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials)
    return redirect("/dashboard")

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@app.route("/dashboard")
def dashboard():
    # If credentials not in session, use sample data
    if 'credentials' not in session:
        return render_template('dashboard.html', 
            steps=get_sample_steps(),
            heart_rate=get_sample_heart_rate(),
            sleep=get_sample_sleep(),
            calories=get_sample_calories()
        )

    # Existing Google Fit API code remains the same
    credentials = Credentials(**session['credentials'])
    service = build('fitness', 'v1', credentials=credentials)

    # Rest of the existing dashboard route code...
    # If no data is retrieved, fall back to sample data
    try:
        # Existing data retrieval and processing code...
        processed = {
            'steps': process_steps(raw_data['steps']),
            'heart_rate': process_heart_rate(raw_data['heart_rate']),
            'sleep': process_sleep(raw_data['sleep']),
            'calories': process_calories(raw_data['calories'])
        }

        return render_template('dashboard.html', 
            steps=processed['steps'],
            heart_rate=processed['heart_rate'],
            sleep=processed['sleep'],
            calories=processed['calories']
        )
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        # Fallback to sample data if API retrieval fails
        return render_template('dashboard.html', 
            steps=get_sample_steps(),
            heart_rate=get_sample_heart_rate(),
            sleep=get_sample_sleep(),
            calories=get_sample_calories()
        )

if __name__ == "__main__":
    app.run(debug=True)