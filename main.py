import argparse
import logging
import cv2
import numpy as np
import json
import threading
from flask import Flask, render_template, Response
import paho.mqtt.client as mqtt
from datetime import datetime
from config.config_manager import load_json_config
from src.calibration import Calibration
import time
import json

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "camera/aruco"

# Flask Configuration
app = Flask(__name__)

# MQTT client setup
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Global variable for the camera feed
cap = None
marker_size = 0.05  # Marker size in meters (adjust as per your setup)



# Variables for MQTT and streaming
last_mqtt_publish_time = 0
annotated_frame = None


def generate_feed():
    """Generates the video stream for Flask."""
    global annotated_frame
    while True:
        if annotated_frame is not None:
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_data = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Route for video stream."""
    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")



def return_camera_indexes():
    index = 0
    arr = []
    i = 5
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def detect_aruco_and_estimate_pose():
    global annotated_frame
    start_time = time.time()
    
    
    global cap
    source = int(load_json_config()['source'])
    cap = cv2.VideoCapture(source)

    # Load camera parameters from configuration
    config = load_json_config()
    camera_matrix = np.array(load_json_config()['mtx'])
    dist_coeffs = np.array(config['dist'])

    # Define ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error('Failed to grab frame from camera')
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(frame)

        marker_info = {}
        len_ids = 0
        distance = 0
        
        if ids is not None:
            len_ids = len(ids)
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Print the number of markers detected
            marker_info["num_markers"] = len(ids)
            marker_info["marker_ids"] = ids.flatten().tolist()

            rvecs, tvecs, trash = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            for rvec, tvec, marker_id in zip(rvecs, tvecs, ids.flatten()):
                # Calculate distance to the camera
                distance = np.linalg.norm(tvec)

                # Draw axis for the marker
                thickness = 5
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03, thickness)
        
        # Update the annotated frame for streaming
        annotated_frame = frame.copy()
                
        # Create a JSON payload
        payload = json.dumps({
            "num_markers": len_ids,
            "distance": round(distance, 2)
        })

        # Publish the payload to the MQTT topic
        client.publish(MQTT_TOPIC, payload)
        print(f"Published: {payload}")

        # Display the frame
        cv2.imshow('Aruco Marker Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description='Aruco Tags Examples',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--aruco', '-a',
        action='store_true', help='Detects Aruco markers and performs pose estimation')

    return parser.parse_args()

if __name__ == '__main__':
    ns = parse_args()

    if ns.aruco:
        # Start a background thread for ArUco marker detection and MQTT publishing
        threading.Thread(target=detect_aruco_and_estimate_pose, daemon=True).start()

    import threading

    # Run detection loop in a separate thread
    detection_thread = threading.Thread(target=detect_aruco_and_estimate_pose)
    detection_thread.daemon = True
    detection_thread.start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000, debug=False)

