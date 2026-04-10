from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "EYEQ Backend Running 🚀"}

# 🔔 Alert API
@app.post("/alert")
def send_alert(data: dict):
    print("🚨 ALERT RECEIVED:", data)
    return {"status": "Alert Sent"}