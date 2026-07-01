import cv2
import numpy as np
import mediapipe as mp
import mss
import time

from config_and_constants import *
from functions import *



# --------------------
# 카메라 초기화 & 보정
# --------------------
cap = cv2.VideoCapture(0)
eye_center_ref, yaw_ref, pitch_ref, roll_ref = calibrate(cap, face_mesh)

prev_face_center = None

with mss.mss() as sct:
    monitor_full = sct.monitors[1]
    monitor = {"top": monitor_full["top"], "left": monitor_full["left"],
               "width": monitor_full["width"], "height": monitor_full["height"]}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, cam = cap.read()
        if ret:
            rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                last_detect_time = time.time()
                lm = res.multi_face_landmarks[0].landmark
                lm_points = np.array([[p.x, p.y] for p in lm])

                # 얼굴 이동 감지
                face_center = np.array([lm[FACE_CENTER].x, lm[FACE_CENTER].y])
                if prev_face_center is not None:
                    movement = np.linalg.norm(face_center - prev_face_center)
                    if movement > MOVEMENT_THRESHOLD:
                        eye_center_ref, yaw_ref, pitch_ref, roll_ref = calibrate(cap, face_mesh)
                prev_face_center = face_center

                # blink 업데이트
                blink_count, avg_blinks = update_blink(lm_points)

                # 시선 좌표 계산
                cursor_x, cursor_y = calculate_gaze(lm, eye_center_ref, yaw_ref, pitch_ref, roll_ref)
        else:
            cursor_x, cursor_y = screen_w//2, screen_h//2

        # ---- 화면 캡처 ----
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ---- 배경 제거 & 객체 감지 ----
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, blur_kernel)
        _, fgmask = cv2.threshold(fgmask, threshold_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            detected_objects.append((cx, cy, w, h))

        objects = update_objects(detected_objects)
        overlay_radius = 300
        tracked_objects = [(cx, cy, obj_id) for obj_id, (cx, cy) in objects.items()
                           if abs(cursor_x - cx) <= overlay_radius and abs(cursor_y - cy) <= overlay_radius]

        # ---- 집중도 계산 ----
        now = time.time()
        focused = len(tracked_objects) > 0
        global fixation_active, fixation_start, focus_score, gaze_out_start

        if avg_blinks * 6.0 >= 61:
            focus_score = 0
        else:
            if focused:
                if not fixation_active:
                    fixation_active = True
                    fixation_start = now
            else:
                if fixation_active:
                    duration = now - fixation_start
                    if duration >= FIXATION_MIN_TIME:
                        focus_score += min(2, duration * 4)
                        focus_durations.append(duration)
                    fixation_active = False

                # 화면 밖 시선 체크
                if (cursor_x + overlay_radius < 0 or cursor_x - overlay_radius > screen_w or
                    cursor_y + overlay_radius < 0 or cursor_y - overlay_radius > screen_h):
                    if gaze_out_start is None:
                        gaze_out_start = time.time()
                    elapsed_out = now - gaze_out_start
                    if elapsed_out >= Gaze_OUT_DURATION:
                        focus_score -= 0.5
                        cv2.putText(frame, "Gaze Out of Screen!", (screen_w//2 - 200, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,   255), 2)
                else:
                    gaze_out_start = None
                    if len(detected_objects) > 0:
                        focus_score -= 0.05
                    else:
                        focus_score -= 0.1

            focus_score = max(0, min(100, focus_score))

        # ---- 시각화 ----
        for (cx, cy, w, h) in detected_objects:
            cv2.rectangle(frame, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (0,255,0), 2)
        for (cx, cy, obj_id) in tracked_objects:
            cv2.circle(frame, (int(cx), int(cy)), 15, (255,0,0), 2)

        frame = draw_eye_overlay(frame, cursor_x, cursor_y, overlay_radius)

        # 집중도 & blink 표시
        cv2.putText(frame, f"Focus: {int(focus_score)}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Blinks(avg10s): {int(avg_blinks)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Blink real-time: {blink_count}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # 졸림 경고
        if avg_blinks * 6.0 >= 61:
            cv2.putText(frame, "Drowsiness detected!", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # 고정 시선 시간
        if fixation_active:
            cv2.putText(frame, f"Fixation: {now - fixation_start:.2f}s", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
