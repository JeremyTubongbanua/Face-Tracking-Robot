#!/usr/bin/env python3
from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np
import RPi.GPIO as GPIO
from time import sleep, time
from flask import Flask, Response, request, render_template_string
import threading

app = Flask(__name__)

SERVO_PIN_1 = 14
SERVO_PIN_2 = 15

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

pwm_servo_1 = GPIO.PWM(SERVO_PIN_1, 50)
pwm_servo_2 = GPIO.PWM(SERVO_PIN_2, 50)

pwm_servo_1.start(0)
pwm_servo_2.start(0)

yaw_angle = 90
pitch_angle = 90

home_yaw = 90
home_pitch = 90

last_detection_time = time()
home_returning = False

kP = 0.1
kI = 0.00001
kD = 0.000001

yaw_error_sum = 0.0
pitch_error_sum = 0.0

yaw_error_prev = 0.0
pitch_error_prev = 0.0

MAX_INTEGRAL = 100.0

frame = None
latest_frame = None
detections = []
frame_lock = threading.Lock()
error_lock = threading.Lock()
pid_lock = threading.Lock()

def set_servo_angle(pwm, angle):
    duty_cycle = (angle / 18.0) + 2
    pwm.ChangeDutyCycle(duty_cycle)
    sleep(0.05)
    pwm.ChangeDutyCycle(0)

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

manip = pipeline.createImageManip()
manip.initialConfig.setResize(300, 300)
manip.initialConfig.setFrameType(depthai.ImgFrame.Type.RGB888p)

detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name='face-detection-retail-0004', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(manip.inputImage)
manip.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

device = depthai.Device(pipeline)
q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def control_servos(center_x, center_y, frame_center_x, frame_center_y):
    global yaw_angle, pitch_angle, home_returning
    global yaw_error_sum, pitch_error_sum
    global yaw_error_prev, pitch_error_prev
    global kP, kI, kD

    if home_returning:
        return

    yaw_error = (frame_center_x - center_x) / frame_center_x
    pitch_error = (frame_center_y - center_y) / frame_center_y

    with error_lock:
        yaw_error_sum += yaw_error
        pitch_error_sum += pitch_error

        yaw_error_sum = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, yaw_error_sum))
        pitch_error_sum = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, pitch_error_sum))

        yaw_error_delta = yaw_error - yaw_error_prev
        pitch_error_delta = pitch_error - pitch_error_prev

        yaw_error_prev = yaw_error
        pitch_error_prev = pitch_error

    if abs(yaw_error) < 0.05 and abs(pitch_error) < 0.05:
        return

    with pid_lock:
        current_kP = kP
        current_kI = kI
        current_kD = kD

    yaw_adjustment = (current_kP * yaw_error +
                      current_kI * yaw_error_sum +
                      current_kD * yaw_error_delta)

    pitch_adjustment = (current_kP * pitch_error +
                        current_kI * pitch_error_sum +
                        current_kD * pitch_error_delta)

    yaw_angle += yaw_adjustment
    pitch_angle += pitch_adjustment

    yaw_angle = max(0, min(180, yaw_angle))
    pitch_angle = max(0, min(180, pitch_angle))

    set_servo_angle(pwm_servo_1, pitch_angle)
    set_servo_angle(pwm_servo_2, yaw_angle)

def return_to_home():
    global yaw_angle, pitch_angle, home_returning
    global yaw_error_sum, pitch_error_sum
    global yaw_error_prev, pitch_error_prev

    if home_returning:
        return
    home_returning = True

    with error_lock:
        yaw_error_sum = 0.0
        pitch_error_sum = 0.0
        yaw_error_prev = 0.0
        pitch_error_prev = 0.0

    set_servo_angle(pwm_servo_1, home_pitch)
    pitch_angle = home_pitch
    set_servo_angle(pwm_servo_2, home_yaw)
    yaw_angle = home_yaw
    home_returning = False

def run_tracking():
    global frame, detections, last_detection_time, home_returning, latest_frame

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            highest_confidence_face = None

            for detection in detections:
                if detection.confidence >= 0.9:
                    if highest_confidence_face is None or detection.confidence > highest_confidence_face.confidence:
                        highest_confidence_face = detection

            current_time = time()

            if highest_confidence_face is not None:
                last_detection_time = current_time
                bbox = frameNorm(frame, (highest_confidence_face.xmin, highest_confidence_face.ymin,
                                         highest_confidence_face.xmax, highest_confidence_face.ymax))

                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2

                cv2.line(frame, (int(center_x), int(center_y)), (int(frame_center_x), int(frame_center_y)), (0, 255, 0), 2)

                threading.Thread(target=control_servos, args=(center_x, center_y, frame_center_x, frame_center_y)).start()

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            else:
                if time() - last_detection_time > 2 and not home_returning:
                    threading.Thread(target=return_to_home).start()

            with frame_lock:
                latest_frame = frame.copy()

def generate_feed():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield b''

        sleep(0.05)

@app.route('/feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tune', methods=['GET', 'POST'])
def tune_pid():
    global kP, kI, kD
    message = ''
    if request.method == 'POST':
        try:
            new_kP = float(request.form.get('kP', kP))
            new_kI = float(request.form.get('kI', kI))
            new_kD = float(request.form.get('kD', kD))
            with pid_lock:
                kP = new_kP
                kI = new_kI
                kD = new_kD
            message = 'PID parameters updated successfully.'
        except ValueError:
            message = 'Invalid input. Please enter numeric values.'
    with pid_lock:
        current_kP = kP
        current_kI = kI
        current_kD = kD
    html = '''
    <!doctype html>
    <title>PID Tuning</title>
    <h1>PID Parameter Tuning</h1>
    <p>{{ message }}</p>
    <form method="post">
        <label for="kP">Proportional Gain (kP):</label><br>
        <input type="text" id="kP" name="kP" value="{{ kP }}"><br><br>
        <label for="kI">Integral Gain (kI):</label><br>
        <input type="text" id="kI" name="kI" value="{{ kI }}"><br><br>
        <label for="kD">Derivative Gain (kD):</label><br>
        <input type="text" id="kD" name="kD" value="{{ kD }}"><br><br>
        <input type="submit" value="Update">
    </form>
    <p>Current kP: {{ kP }}</p>
    <p>Current kI: {{ kI }}</p>
    <p>Current kD: {{ kD }}</p>
    <p><a href="/feed">View Video Feed</a></p>
    '''
    return render_template_string(html, kP=current_kP, kI=current_kI, kD=current_kD, message=message)

if __name__ == "__main__":
    tracking_thread = threading.Thread(target=run_tracking)
    tracking_thread.daemon = True
    tracking_thread.start()

    app.run(host='0.0.0.0', port=5000)

    pwm_servo_1.stop()
    pwm_servo_2.stop()
    GPIO.cleanup()
