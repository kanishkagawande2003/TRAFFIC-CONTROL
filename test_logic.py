import streamlit as st
import cv2
from ultralytics import YOLO
import requests
import time

# --- CONFIG ---
ESP_IP = "http://10.20.124.109" # Update this from Serial Monitor

st.set_page_config(page_title="AI Traffic & Safety", layout="wide")
st.title("🚦 AI Traffic Signal & Helmet Detection")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt") # Ensure this model has 'helmet' and 'person' classes

model = load_yolo()

def send_request(params):
    try:
        requests.get(f"{ESP_IP}/traffic", params=params, timeout=0.6)
    except:
        pass

def get_cooldown(count):
    if count <= 3: return 23
    elif count <= 6: return 26
    else: return 30

# --- UI ---
col1, col2 = st.columns([3, 1])
st_frame = col1.empty()
st_info = col2.empty()

cap = cv2.VideoCapture(1) # Change to 0 for internal webcam
last_traffic_time = 0

if st.sidebar.button("Start System"):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        vehicle_count = 0
        person_detected = False
        helmet_detected = False
        
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Traffic Counting Logic
                if label in ["car", "motorcycle", "bus", "truck"]:
                    vehicle_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Helmet Detection Logic
                if label == "person":
                    person_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if label == "helmet":
                    helmet_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Trigger Buzzer if person is found but no helmet
        if person_detected and not helmet_detected:
            cv2.putText(frame, "!!! NO HELMET !!!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            send_request({"alert": "1"})

        st_frame.image(frame, channels="BGR")

        # Traffic Timer Logic
        curr_time = time.time()
        wait_time = get_cooldown(vehicle_count)
        time_left = int(wait_time - (curr_time - last_traffic_time))

        with st_info.container():
            st.metric("Vehicles", vehicle_count)
            if time_left <= 0:
                st.success("🔄 Updating Signal...")
                send_request({"count": vehicle_count})
                last_traffic_time = time.time()
            else:
                st.info(f"Signal active: {time_left}s")
            
            if person_detected and not helmet_detected:
                st.error("🚨 HELMET VIOLATION")

    cap.release()