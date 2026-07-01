# 1. 초기값 및 인덱스 설정 파일

import mediapipe as mp
import cv2
import time
import numpy as np

# ————————————————————————
# Mediapipe 초기화
# ————————————————————————

mp_face_mesh = mp.solutions.face_mesh
# FaceMesh 객체는 메인에서 import 후 사용될 수 있도록 초기화는 여기에 둠.
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ————————————————————————
# Landmarks 인덱스 (상수)
# ————————————————————————
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
FACE_CENTER = 1
LEFT_FACE = 234
RIGHT_FACE = 454

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ————————————————————————
# 집중도 & Tracking 설정 (초기값)
# ————————————————————————
focus_score = 100
focus_durations = []
fixation_start = None
fixation_active = False
FIXATION_MIN_TIME = 0.25 # 상수

# ————————————————————————
# 깜빡임 설정 (초기값)
# ————————————————————————
EAR_THRESHOLD = 0.25 # 상수
blink_count = 0
avg_blinks = 0
blink_start_time = time.time()

# ————————————————————————
# 배경 제거 (초기값)
# ————————————————————————
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)
min_contour_area = 500 # 상수
blur_kernel = 7 # 상수
threshold_val = 150 # 상수

# ————————————————————————
# 시선 계산 관련 가중치 (상수)
# ————————————————————————
SENSITIVITY_EYE_X = 50.0
SENSITIVITY_EYE_Y = 40.0
SENSITIVITY_FACE = 0.1
SMOOTH_ALPHA = 0.8
LOST_TIME = 2.0
MOVEMENT_THRESHOLD = 0.025

# ————————————————————————
# 화면 관련 (상수)
# ————————————————————————
screen_w, screen_h = 1920, 1080

# ————————————————————————
# Centroid tracking 변수 (초기값)
# ————————————————————————
next_object_id = 0
objects = {}  # {id: (cx, cy)}

WINDOW_NAME = "Eye Tracking Focus" # 상수

# ————————————————————————
# 시선 원 화면 밖 체크 변수 (초기값)
# ————————————————————————
gaze_out_start = None
Gaze_OUT_DURATION = 3.0  # 3초 이상 화면 밖이면 점수 감소 (상수)

# 메인 루프 변수 초기화 (메인 루프 시작 전 초기화 필요)
smoothed_dx, smoothed_dy = 0, 0
prev_face_center = None
last_detect_time = time.time()