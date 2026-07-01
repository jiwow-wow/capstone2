import numpy as np
import cv2
import time
from scipy.spatial import distance
from config_and_constants import *

# --------------------
# 유틸 함수
# --------------------
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(landmarks, eye_indices):
    A = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

def update_blink(lm_list):
    global blink_count, avg_blinks, blink_start_time

    if lm_list is None:
        return blink_count, avg_blinks

    landmarks_xy = np.array([[p[0], p[1]] for p in lm_list])
    left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX)
    right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX)
    ear = (left_ear + right_ear) / 2.0

    if ear < EAR_THRESHOLD:
        blink_count += 1

    elapsed_time = time.time() - blink_start_time
    if elapsed_time >= 10:
        avg_blinks = blink_count / (elapsed_time / 10)
        blink_count = 0
        blink_start_time = time.time()

    return blink_count, avg_blinks

def draw_eye_overlay(frame, cursor_x, cursor_y, radius):
    cv2.circle(frame, (int(cursor_x), int(cursor_y)), radius, (0, 255, 0), 2)
    return frame

def calibrate(cap, face_mesh, duration=2, message="Look at center"):
    start_time = time.time()
    eye_samples, yaw_samples, pitch_samples, roll_samples = [], [], [], []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_FULLSCREEN)

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_iris = np.mean([lm[i].x for i in LEFT_IRIS]), np.mean([lm[i].y for i in LEFT_IRIS])
            right_iris = np.mean([lm[i].x for i in RIGHT_IRIS]), np.mean([lm[i].y for i in RIGHT_IRIS])
            avg_eye = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)
            eye_samples.append(avg_eye)

            dx_face = lm[RIGHT_FACE].x - lm[LEFT_FACE].x or 0.001
            yaw = (lm[FACE_CENTER].x - lm[LEFT_FACE].x) / dx_face
            pitch = lm[FACE_CENTER].y - (lm[LEFT_EYE_TOP].y + lm[RIGHT_EYE_TOP].y) / 2
            roll = lm[LEFT_EYE_TOP].y - lm[RIGHT_EYE_TOP].y

            yaw_samples.append(yaw)
            pitch_samples.append(pitch)
            roll_samples.append(roll)

        cv2.putText(frame, message, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    return (
        np.mean([e[0] for e in eye_samples]),
        np.mean([e[1] for e in eye_samples]),
    ), np.mean(yaw_samples), np.mean(pitch_samples), np.mean(roll_samples)

def calculate_gaze(lm, eye_center_ref, yaw_ref, pitch_ref, roll_ref):
    left_iris = np.mean([lm[i].x for i in LEFT_IRIS]), np.mean([lm[i].y for i in LEFT_IRIS])
    right_iris = np.mean([lm[i].x for i in RIGHT_IRIS]), np.mean([lm[i].y for i in RIGHT_IRIS])
    avg_eye = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)

    dx_face = lm[RIGHT_FACE].x - lm[LEFT_FACE].x or 0.001
    yaw = (lm[FACE_CENTER].x - lm[LEFT_FACE].x) / dx_face
    pitch = lm[FACE_CENTER].y - (lm[LEFT_EYE_TOP].y + lm[RIGHT_EYE_TOP].y)/2
    roll = lm[LEFT_EYE_TOP].y - lm[RIGHT_EYE_TOP].y

    dx_eye = (avg_eye[0] - eye_center_ref[0]) * SENSITIVITY_EYE_X
    dy_eye = (avg_eye[1] - eye_center_ref[1]) * SENSITIVITY_EYE_Y
    dx_face_c = (yaw_ref - yaw - 0.5 * roll) * SENSITIVITY_FACE
    dy_face_c = (pitch_ref - pitch) * SENSITIVITY_FACE

    global smoothed_dx, smoothed_dy
    dx = -(dx_eye + dx_face_c)
    dy = dy_eye + dy_face_c
    smoothed_dx = SMOOTH_ALPHA * dx + (1 - SMOOTH_ALPHA) * smoothed_dx
    smoothed_dy = SMOOTH_ALPHA * dy + (1 - SMOOTH_ALPHA) * smoothed_dy

    cursor_x = int(screen_w/2 + smoothed_dx * screen_w)
    cursor_y = int(screen_h/2 + smoothed_dy * screen_h)

    return cursor_x, cursor_y

def update_objects(detected_objects):
    global next_object_id, objects
    new_objects = {}

    if len(objects) == 0:
        for (cx, cy, w, h) in detected_objects:
            new_objects[next_object_id] = (cx, cy)
            next_object_id += 1
    else:
        object_ids = list(objects.keys())
        object_centroids = np.array(list(objects.values()))
        detected_centroids = np.array([[cx, cy] for (cx, cy, w, h) in detected_objects])

        if len(detected_centroids) == 0:
            objects.clear()
            return objects

        D = distance.cdist(object_centroids, detected_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_detections = set()
        for r, c in zip(rows, cols):
            if c in assigned_detections:
                continue
            new_objects[object_ids[r]] = tuple(detected_centroids[c])
            assigned_detections.add(c)

        for i, (cx, cy, w, h) in enumerate(detected_objects):
            if i not in assigned_detections:
                new_objects[next_object_id] = (cx, cy)
                next_object_id += 1

    objects = new_objects.copy()
    return objects
