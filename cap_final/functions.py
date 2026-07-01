# 2. 함수 정의 파일

import cv2
import numpy as np
import time
from scipy.spatial import distance
from config_and_constants import *

# ————————————————————————
# Utility 함수
# ————————————————————————

def euclidean_distance(a, b):
    # a와 b는 (x, y) 형태의 리스트/튜플 또는 numpy 배열
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(landmarks, eye_indices):
    # 눈꺼풀 상하 거리: 수직 거리 A, B
    A = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    # 눈꼬리/눈머리 거리: 수평 거리 C
    C = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

# ————————————————————————
# 깜빡임 함수
# ————————————————————————

def update_blink(lm_list):
    """
    깜빡임을 감지하고 10초당 평균 깜빡임 횟수를 업데이트합니다.
    """
    global blink_count, avg_blinks, blink_start_time

    # 현재 EAR 계산
    left_ear = eye_aspect_ratio(lm_list, LEFT_EYE_IDX)
    right_ear = eye_aspect_ratio(lm_list, RIGHT_EYE_IDX)
    ear = (left_ear + right_ear) / 2.0

    # 깜빡임 감지
    if ear < EAR_THRESHOLD:
        blink_count += 1

    # 10초마다 평균 깜빡임 업데이트
    elapsed_time = time.time() - blink_start_time
    if elapsed_time >= 10:
        avg_blinks = blink_count / (elapsed_time / 10) # 10초당 평균으로 변환
        blink_count = 0
        blink_start_time = time.time()

    return blink_count, avg_blinks

# -------------------------------------------------
# 시각화 함수
# -------------------------------------------------

def draw_eye_overlay(frame, cursor_x, cursor_y, radius):
    """
    시선 위치에 초점 원을 그립니다.
    """
    # 원이 화면 밖에 있을 때 튀는 현상을 방지하기 위해 경계 체크
    if 0 <= cursor_x < screen_w and 0 <= cursor_y < screen_h:
        cv2.circle(frame, (int(cursor_x), int(cursor_y)), radius, (0, 255, 0), 2)
    return frame

# -------------------------------------------------
# 보정 및 시선 계산 함수
# -------------------------------------------------

def calibrate(cap, face_mesh, duration=2, message="Look at center"):
    """
    초기 보정: 사용자가 화면 중앙을 볼 때의 눈과 얼굴 기준값을 측정합니다.
    """
    start_time = time.time()
    eye_samples, yaw_samples, pitch_samples, roll_samples = [], [], [], []

    # 보정 메시지 표시를 위해 임시 윈도우 생성 (메인 루프에서 윈도우 이름 재사용)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_FULLSCREEN)

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # 눈동자 중심 측정
            left_iris = np.mean([lm[i].x for i in LEFT_IRIS]), np.mean([lm[i].y for i in LEFT_IRIS])
            right_iris = np.mean([lm[i].x for i in RIGHT_IRIS]), np.mean([lm[i].y for i in RIGHT_IRIS])
            avg_eye = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)
            eye_samples.append(avg_eye)

            # 얼굴 각도 (Yaw, Pitch, Roll) 측정
            dx_face = lm[RIGHT_FACE].x - lm[LEFT_FACE].x or 0.001
            yaw = (lm[FACE_CENTER].x - lm[LEFT_FACE].x) / dx_face
            pitch = lm[FACE_CENTER].y - (lm[LEFT_EYE_TOP].y + lm[RIGHT_EYE_TOP].y) / 2
            roll = lm[LEFT_EYE_TOP].y - lm[RIGHT_EYE_TOP].y

            yaw_samples.append(yaw)
            pitch_samples.append(pitch)
            roll_samples.append(roll)

        # 보정 메시지 표시
        cv2.putText(frame, message, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC로 종료
            break

    # 평균 보정값 반환
    return (
        np.mean([e[0] for e in eye_samples]),
        np.mean([e[1] for e in eye_samples]),
    ), np.mean(yaw_samples), np.mean(pitch_samples), np.mean(roll_samples)


def calculate_gaze(lm, eye_center_ref, yaw_ref, pitch_ref, roll_ref):
    """
    보정된 기준값을 사용하여 현재의 시선(gaze) 위치를 계산합니다.
    """
    global smoothed_dx, smoothed_dy
    
    # 1. 현재 눈동자 위치 측정
    left_iris = np.mean([lm[i].x for i in LEFT_IRIS]), np.mean([lm[i].y for i in LEFT_IRIS])
    right_iris = np.mean([lm[i].x for i in RIGHT_IRIS]), np.mean([lm[i].y for i in RIGHT_IRIS])
    avg_eye = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)

    # 2. 현재 얼굴 각도 측정
    dx_face = lm[RIGHT_FACE].x - lm[LEFT_FACE].x or 0.001
    yaw = (lm[FACE_CENTER].x - lm[LEFT_FACE].x) / dx_face
    pitch = lm[FACE_CENTER].y - (lm[LEFT_EYE_TOP].y + lm[RIGHT_EYE_TOP].y)/2
    roll = lm[LEFT_EYE_TOP].y - lm[RIGHT_EYE_TOP].y

    # 3. 눈동자 움직임에 의한 변화량
    dx_eye = (avg_eye[0] - eye_center_ref[0]) * SENSITIVITY_EYE_X
    dy_eye = (avg_eye[1] - eye_center_ref[1]) * SENSITIVITY_EYE_Y
    
    # 4. 얼굴 움직임에 의한 변화량
    dx_face_c = (yaw_ref - yaw - 0.5 * roll) * SENSITIVITY_FACE
    dy_face_c = (pitch_ref - pitch) * SENSITIVITY_FACE

    # 5. 전체 변화량
    dx = -(dx_eye + dx_face_c) # 좌우(X)
    dy = dy_eye + dy_face_c # 상하(Y)

    # 6. 부드럽게 만들기 (Smoothing)
    smoothed_dx = SMOOTH_ALPHA * dx + (1 - SMOOTH_ALPHA) * smoothed_dx
    smoothed_dy = SMOOTH_ALPHA * dy + (1 - SMOOTH_ALPHA) * smoothed_dy

    # 7. 화면 좌표로 변환
    cursor_x = int(screen_w/2 + smoothed_dx * screen_w)
    cursor_y = int(screen_h/2 + smoothed_dy * screen_h)

    return cursor_x, cursor_y

# -------------------------------------------------
# Tracking 함수
# -------------------------------------------------

def update_objects(detected_objects):
    """
    이전 프레임의 객체와 현재 프레임에서 감지된 객체를 연결(추적)합니다.
    (Centroid Tracking의 단순 구현)
    """
    global next_object_id, objects
    new_objects = {}

    if len(objects) == 0:
        # 이전에 객체가 없었으면, 새 ID 부여
        for (cx, cy, w, h) in detected_objects:
            new_objects[next_object_id] = (cx, cy)
            next_object_id += 1
    else:
        object_ids = list(objects.keys())
        object_centroids = np.array(list(objects.values()))
        # 감지된 객체의 중심점 (cx, cy)만 추출
        detected_centroids = np.array([[cx, cy] for (cx, cy, w, h) in detected_objects])

        if len(detected_centroids) == 0:
            # 감지된 객체가 없으면, 기존 객체 목록 초기화
            objects.clear()
            return objects

        # 기존 객체와 감지된 객체 간의 유클리드 거리 행렬 계산
        D = distance.cdist(object_centroids, detected_centroids)
        # 최소 거리를 기준으로 연결 (간단한 최단 거리 매칭)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_detections = set()
        for r, c in zip(rows, cols):
            if c in assigned_detections:
                continue
            # 기존 ID에 새로운 위치 할당
            new_objects[object_ids[r]] = tuple(detected_centroids[c])
            assigned_detections.add(c)

        # 매칭되지 않은(새로운) 감지된 객체에 새 ID 부여
        for i, (cx, cy, w, h) in enumerate(detected_objects):
            if i not in assigned_detections:
                new_objects[next_object_id] = (cx, cy)
                next_object_id += 1

    objects = new_objects.copy()
    return objects