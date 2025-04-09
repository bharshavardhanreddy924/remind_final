from datetime import datetime
import time
from pymongo import MongoClient
import certifi
import requests
import json

# MongoDB Configuration
uri = "mongodb+srv://bharshavardhanreddy924:516374Ta@data-dine.5oghq.mongodb.net/?retryWrites=true&w=majority&ssl=true"

try:
    client = MongoClient(uri, tlsCAFile=certifi.where())
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")
    db = client['remind_db']
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None

def process_notifications():
    """Process and send scheduled notifications"""
    if not db:
        print("Database connection not available")
        return

    while True:
        try:
            # Get current time
            current_time = datetime.now()
            
            # Find notifications that are due
            notifications = db.notifications.find({
                "scheduled_time": {"$lte": current_time},
                "sent": {"$ne": True}
            })
            
            for notification in notifications:
                try:
                    # Get user's push subscription
                    user = db.users.find_one({"_id": notification["user_id"]})
                    if not user or not user.get("push_subscription"):
                        continue
                    
                    push_subscription = user["push_subscription"]
                    
                    # Prepare notification payload
                    payload = {
                        "title": notification["title"],
                        "body": notification["body"],
                        "icon": "/static/images/icons/icon-192x192.svg",
                        "badge": "/static/images/icons/icon-192x192.svg",
                        "vibrate": [100, 50, 100],
                        "data": {
                            "type": notification["type"],
                            "dateOfArrival": current_time.isoformat(),
                            "primaryKey": str(notification["_id"])
                        },
                        "actions": [
                            {
                                "action": "explore",
                                "title": "View Details",
                                "icon": "/static/images/icons/icon-192x192.svg"
                            },
                            {
                                "action": "close",
                                "title": "Close",
                                "icon": "/static/images/icons/icon-192x192.svg"
                            }
                        ]
                    }
                    
                    # Send push notification
                    response = requests.post(
                        push_subscription["endpoint"],
                        json=payload,
                        headers={
                            "Authorization": f"key={push_subscription['keys']['auth']}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 201:
                        # Mark notification as sent
                        db.notifications.update_one(
                            {"_id": notification["_id"]},
                            {"$set": {"sent": True}}
                        )
                        print(f"Notification sent successfully: {notification['title']}")
                    else:
                        print(f"Failed to send notification: {response.status_code}")
                        
                except Exception as e:
                    print(f"Error processing notification: {e}")
                    continue
            
            # Sleep for 1 minute before checking again
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in notification processor: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    print("Starting notification processor...")
    process_notifications() 