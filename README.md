# MemoryCare - Memory Support Flask Application

MemoryCare is a comprehensive web application designed to help individuals with memory impairments and their caregivers. The application provides tools for memory training, medication management, task scheduling, note-taking, and a personalized AI assistant.

## Features

- **User & Caretaker Roles**: Different interfaces and capabilities for patients and caregivers
- **Task Management**: Create, track, and complete daily tasks
- **Medication Reminders**: Schedule and manage medication times
- **Memory Training Games**: Interactive games to exercise memory skills
- **Personal Notes**: Record and access important information
- **AI Memory Assistant**: Personalized AI that answers questions based on individual information
- **Caretaker Dashboard**: Allows caregivers to manage multiple patients and their care

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: MongoDB Atlas
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **NLP**: NLTK and scikit-learn for the AI assistant

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account

### Installation

1. Clone the repository
   ```
   git clone <repository-url>
   cd memorycare
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   # For Windows
   venv\Scripts\activate
   # For macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables
   - The MongoDB URI is already included in the app.py file
   - In production, you should use environment variables instead of hardcoding the URI

5. Run the application
   ```
   python app.py
   ```

6. Access the application at `http://localhost:5000`

## User Guide

### For Patients

1. **Registration**: Create an account with the "user" role. Optionally, add your caretaker's email.
2. **Dashboard**: Access all features from your personalized dashboard.
3. **Tasks**: Create and manage your daily tasks.
4. **Medications**: Set up medication reminders with timing.
5. **Memory Training**: Play memory games to strengthen your memory skills.
6. **Notes**: Keep track of important information.
7. **AI Assistant**: Ask questions about your personal information.

### For Caretakers

1. **Registration**: Create an account with the "caretaker" role.
2. **Dashboard**: See a list of all patients under your care.
3. **Patient Management**: Click "Manage" to access a specific patient's information.
4. **Update Personal Info**: Add detailed personal information that will be used by the AI assistant.
5. **Manage Tasks**: Create and monitor tasks for your patients.
6. **Manage Medications**: Set up medication schedules for your patients.
7. **Add Notes**: Keep notes on patient care and progress.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [Bootstrap](https://getbootstrap.com/) for UI components
- [Font Awesome](https://fontawesome.com/) for icons
- [NLTK](https://www.nltk.org/) for natural language processing
- [scikit-learn](https://scikit-learn.org/) for NLP algorithms 