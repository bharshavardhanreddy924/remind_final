from app3 import app

if __name__ == "__main__":
    from app_config import PORT, HOST, DEBUG
    app.run(host=HOST, port=PORT, debug=DEBUG) 