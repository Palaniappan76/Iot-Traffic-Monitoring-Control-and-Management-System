import cv2
import serial
import time
import sys
import requests
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime

# --- CONFIGURATION ---
VIDEO_SOURCE = 0 
SERIAL_PORT = 'COM20' 
BAUD_RATE = 9600
ENABLE_SERIAL = False # Set to True if Arduino is connected for traffic lights

# Traffic Thresholds
THRESHOLD_MEDIUM = 5
THRESHOLD_HIGH = 10
# --- ESP32 CONFIGURATION ---
ESP32_IP = "10.87.216.76"  # REPLACE with your ESP32 IP Address
ESP32_PORT = "80"
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}/update"
ESP32_ACCIDENT_URL = f"http://{ESP32_IP}:{ESP32_PORT}/accident"
ESP32_INTERVAL = 5  # Seconds

# --- EMAIL CONFIGURATION --- egbd hvhv zukr jdwc
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "projectpurposes982005@gmail.com"       # REPLACE with your Gmail
SENDER_PASSWORD = "lrbvwzispuuuhwkc"       # REPLACE with Gmail App Password
RECEIVER_EMAIL = "palanisenthil7667@gmail.com" # REPLACE with Receiver Email
EMAIL_INTERVAL_UPDATES = 4 # Send email after collecting 16 updates (approx 4 mins)

# --- GLOBAL VARIABLES ---
traffic_log = [] # Stores data for email report
last_esp32_time = 0
last_accident_time = 0
ACCIDENT_COOLDOWN = 60 # Seconds to wait before sending another accident alert
accident_active = False  # Tracks whether accident banner should be shown on screen
email_lock = threading.Lock()

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt') 
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize Serial
arduino = None
if ENABLE_SERIAL:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
        ENABLE_SERIAL = False

# --- HELPER FUNCTIONS ---

def get_density_status(count):
    if count < THRESHOLD_MEDIUM:
        return 'L', (0, 255, 0), "LOW"
    elif count < THRESHOLD_HIGH:
        return 'M', (0, 255, 255), "MEDIUM"
    else:
        return 'H', (0, 0, 255), "HIGH"

def check_esp32_connection():
    """Checks if ESP32 is reachable. Stops program if not."""
    print(f"Checking connection to ESP32 at {ESP32_IP}...")
    try:
        # Try to get the main page with a short timeout
        response = requests.get(f"http://{ESP32_IP}:{ESP32_PORT}/", timeout=5)
        if response.status_code == 200:
            print("✅ ESP32 Web Server is Active.")
            return True
        else:
            print(f"❌ ESP32 returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ CRITICAL ERROR: Cannot connect to ESP32 Web Server. {e}")
        print("Execution stopped as per configuration.")
        return False

def send_to_esp32(count, density_text):
    """Sends data to ESP32 Web Server"""
    try:
        payload = {
            'count': str(count),
            'density': density_text,
            'time': datetime.now().strftime("%H:%M:%S")
        }
        requests.post(ESP32_URL, data=payload, timeout=2)
    except Exception as e:
        print(f"⚠️ Failed to send to ESP32: {e}")

def send_accident_to_esp32(date_str, time_str):
    """Sends accident event to ESP32 /accident endpoint for storage on web server"""
    try:
        payload = {
            'date': date_str,
            'time': time_str,
            'event': 'ACCIDENT DETECTED'
        }
        requests.post(ESP32_ACCIDENT_URL, data=payload, timeout=2)
        print(f"📡 Accident event sent to ESP32 web server at {time_str}.")
    except Exception as e:
        print(f"⚠️ Failed to send accident to ESP32: {e}")

def send_email_report(log_data):
    """Sends the 4-minute traffic report via Gmail"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f"Traffic Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Calculate Statistics
        total_vehicles = sum(entry['count'] for entry in log_data)
        avg_vehicles = total_vehicles / len(log_data) if log_data else 0
        
        # Determine Dominant Density
        density_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for entry in log_data:
            density_counts[entry['density']] += 1
        dominant_density = max(density_counts, key=density_counts.get)

        # Build Email Body
        body = f"<h2>🚦 Smart Traffic Monitor Report</h2>"
        body += f"<p><strong>Report Duration:</strong> 4 Minutes (16 Updates)</p>"
        body += f"<p><strong>Average Vehicle Count:</strong> {avg_vehicles:.2f}</p>"
        body += f"<p><strong>Dominant Density:</strong> {dominant_density}</p>"
        body += "<h3>📊 Detailed Log (16 Updates)</h3>"
        body += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
        body += "<tr><th>Time</th><th>Count</th><th>Density</th></tr>"
        
        for entry in log_data:
            color = "green" if entry['density'] == "LOW" else "orange" if entry['density'] == "MEDIUM" else "red"
            body += f"<tr><td>{entry['time']}</td><td>{entry['count']}</td><td style='color:{color}; font-weight:bold;'>{entry['density']}</td></tr>"
        
        body += "</table>"
        body += "<br><p><i>Generated by  Smart Monitor</i></p>"

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email Report Sent Successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def email_thread_worker(log_data):
    """Wrapper to run email sending in a separate thread"""
    thread = threading.Thread(target=send_email_report, args=(log_data,))
    thread.start()

def check_accident(boxes):
    """Basic collision heuristic: checks if vehicle bounding boxes have significant overlap."""
    if len(boxes) < 2:
        return False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i].xyxy[0].cpu().numpy()
            box2 = boxes[j].xyxy[0].cpu().numpy()
            
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                continue
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            min_area = min(box1_area, box2_area)
            if min_area > 0 and (intersection_area / min_area) > 0.6:
                return True
    return False

def send_accident_alert(frame, date_str, time_str):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f"🚨 URGENT: Accident Detected at {time_str}"

        body = f"<h2>⚠️ Accident Detection Alert</h2>"
        body += "<p>An accident has been detected by the Smart Traffic Monitor.</p>"
        body += f"<p><strong>Date:</strong> {date_str}</p>"
        body += f"<p><strong>Time:</strong> {time_str}</p>"
        body += "<p>Please find the captured image attached for reference.</p>"
        
        msg.attach(MIMEText(body, 'html'))

        # Encode frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            image_attachment = MIMEImage(buffer.tobytes())
            image_attachment.add_header('Content-Disposition', 'attachment; filename="accident_capture.jpg"')
            msg.attach(image_attachment)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Accident Alert Email Sent Successfully at {time_str}.")
    except Exception as e:
        print(f"❌ Failed to send accident email: {e}")

def accident_thread_worker(frame_copy, date_str, time_str):
    thread = threading.Thread(target=send_accident_alert, args=(frame_copy, date_str, time_str))
    thread.start()

# --- MAIN EXECUTION ---

def main():
    global traffic_log, last_esp32_time, last_accident_time, accident_active

    # 1. Critical Check: ESP32 Connectivity
    if not check_esp32_connection():
        sys.exit(1) # Stop execution if ESP32 is offline

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    last_signal_time = 0
    signal_interval = 2 

    print("Starting Traffic Monitor... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False)
        vehicle_count = len(results[0].boxes)
        status_code, color, status_text = get_density_status(vehicle_count)

        # --- FEATURE 1: ESP32 UPDATE (Every 15 Seconds) ---
        current_time = time.time()
        if current_time - last_esp32_time >= ESP32_INTERVAL:
            # Send to ESP32
            send_to_esp32(vehicle_count, status_text)
            
            # Add to Log for Email
            traffic_log.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'count': vehicle_count,
                'density': status_text
            })
            
            last_esp32_time = current_time
            print(f"📡 Data sent to ESP32 | Count: {vehicle_count} | Density: {status_text}")

            # --- FEATURE 2: EMAIL REPORT (Every 16 Updates) ---
            if len(traffic_log) >= EMAIL_INTERVAL_UPDATES:
                print("📧 Sending 4-minute email report...")
                # Send email in background thread to avoid freezing video
                email_thread_worker(traffic_log.copy())
                traffic_log.clear() # Reset log for next 4 minutes

        # --- Original Serial Logic ---
        if ENABLE_SERIAL and (time.time() - last_signal_time > signal_interval):
            arduino.write(status_code.encode())
            last_signal_time = time.time()

        # --- Visualization ---
        annotated_frame = results[0].plot()

        # --- FEATURE 3: ACCIDENT DETECTION ---
        current_time = time.time()
        accident_now = check_accident(results[0].boxes)
        if accident_now:
            accident_active = True
            if (current_time - last_accident_time) > ACCIDENT_COOLDOWN:
                print("⚠️ ACCIDENT DETECTED! Triggering alert...")
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                # Send email alert (with frame snapshot) in background thread
                alert_frame = annotated_frame.copy()
                cv2.putText(alert_frame, "ACCIDENT DETECTED!", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                accident_thread_worker(alert_frame, date_str, time_str)

                # Notify ESP32 web server
                threading.Thread(target=send_accident_to_esp32, args=(date_str, time_str), daemon=True).start()

                last_accident_time = current_time
        else:
            # Clear the on-screen banner after 5 seconds of no detection
            if accident_active and (current_time - last_accident_time) > 5:
                accident_active = False

        # --- HUD Background Panel ---
        panel_height = 185 if accident_active else 160
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        cv2.putText(annotated_frame, "TRAFFIC MONITOR", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Density : {status_text}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- ACCIDENT BANNER on live display ---
        if accident_active:
            # Flashing red rectangle across full width
            if int(current_time * 2) % 2 == 0:  # blink every 0.5 s
                cv2.rectangle(annotated_frame, (0, 130), (annotated_frame.shape[1], 185), (0, 0, 200), -1)
            else:
                cv2.rectangle(annotated_frame, (0, 130), (annotated_frame.shape[1], 185), (0, 0, 120), -1)
            cv2.putText(annotated_frame, "🚨 ACCIDENT DETECTED!", (20, 170),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Smart Traffic Density Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()