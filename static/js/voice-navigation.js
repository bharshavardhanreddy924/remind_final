// voice-navigation.js - Place this in your static/js directory

class VoiceNavigation {
    constructor() {
        // Check if browser supports speech recognition
        this.recognition = null;
        this.isListening = false;
        this.statusElement = null;
        this.feedbackElement = null;
        this.voiceButtonElement = null;
        this.synth = window.speechSynthesis;
        
        // Feature detection
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.setupRecognition();
        }
    }
    
    setupRecognition() {
        // Configure recognition settings
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        
        // Handle results
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript.trim();
            const confidence = event.results[0][0].confidence;
            
            console.log(`Recognized: "${transcript}" (Confidence: ${confidence.toFixed(2)})`);
            this.updateStatus(`Recognized: "${transcript}"`);
            
            // Process the command
            this.processCommand(transcript);
        };
        
        // Handle errors
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.updateStatus(`Error: ${event.error}`);
            this.stopListening();
        };
        
        // Handle end of recognition
        this.recognition.onend = () => {
            this.stopListening();
        };
    }
    
    // Initialize the UI components
    initUI(buttonId = 'voiceNavButton', statusId = 'voiceStatus', feedbackId = 'voiceFeedback') {
        this.voiceButtonElement = document.getElementById(buttonId);
        this.statusElement = document.getElementById(statusId);
        this.feedbackElement = document.getElementById(feedbackId);
        
        if (!this.voiceButtonElement) {
            console.error('Voice button element not found!');
            return false;
        }
        
        // Check if speech recognition is supported
        if (!this.recognition) {
            this.voiceButtonElement.disabled = true;
            this.voiceButtonElement.title = 'Voice navigation not supported in this browser';
            this.updateStatus('Voice navigation not supported in your browser.');
            return false;
        }
        
        // Set up event listener for the button
        this.voiceButtonElement.addEventListener('click', () => {
            if (this.isListening) {
                this.stopListening();
            } else {
                this.startListening();
            }
        });
        
        return true;
    }
    
   // Start listening for voice commands
startListening() {
    if (!this.recognition) return;
    
    try {
        // Request microphone permission explicitly before starting
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(() => {
                this.recognition.start();
                this.isListening = true;
                this.updateStatus('Listening... Say a command like "Go to dashboard" or "Tasks"');
                this.updateButtonState(true);
            })
            .catch(error => {
                console.error('Microphone permission denied:', error);
                this.updateStatus('Microphone access denied. Please allow microphone access in your browser settings.');
            });
    } catch (error) {
        console.error('Failed to start speech recognition:', error);
        this.updateStatus('Failed to start listening. Please try again.');
    }
}
    
    // Stop listening for voice commands
    stopListening() {
        if (!this.recognition) return;
        
        try {
            this.recognition.stop();
        } catch (error) {
            console.error('Error stopping recognition:', error);
        }
        
        this.isListening = false;
        this.updateButtonState(false);
    }
    
    // Process the recognized command
    processCommand(command) {
        fetch('/api/process_voice_command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ command: command }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Command processing result:', data);
            
            if (data.status === 'success' && data.redirect) {
                // Provide feedback before redirecting
                this.provideFeedback(data.message || 'Navigating...', () => {
                    window.location.href = data.redirect;
                });
            } else {
                // Just provide feedback for errors or non-navigation commands
                this.provideFeedback(data.message || 'Command not recognized.');
            }
        })
        .catch(error => {
            console.error('Error processing command:', error);
            this.provideFeedback('Error processing your command. Please try again.');
        });
    }
    
    // Provide audio and visual feedback
    provideFeedback(message, callback = null) {
        // Update visual feedback
        if (this.feedbackElement) {
            this.feedbackElement.textContent = message;
        }
        
        // Provide audio feedback using text-to-speech
        if (this.synth && this.synth.speaking) {
            this.synth.cancel(); // Stop any ongoing speech
        }
        
        if (this.synth) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            if (callback) {
                utterance.onend = callback;
            }
            
            this.synth.speak(utterance);
        } else if (callback) {
            // If speech synthesis is not available, still execute the callback after a delay
            setTimeout(callback, 1000);
        }
    }
    
    // Update the status message
    updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
    }
    
    // Update the button state (visual feedback)
    updateButtonState(isListening) {
        if (!this.voiceButtonElement) return;
        
        if (isListening) {
            this.voiceButtonElement.classList.add('listening');
            this.voiceButtonElement.setAttribute('aria-pressed', 'true');
        } else {
            this.voiceButtonElement.classList.remove('listening');
            this.voiceButtonElement.setAttribute('aria-pressed', 'false');
        }
    }
}

// Initialize when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const voiceNav = new VoiceNavigation();
    const initialized = voiceNav.initUI();
    
    if (initialized) {
        console.log('Voice navigation initialized successfully');
    }
});