#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import yaml
from dynamixel_sdk import *
from ultralytics import YOLO
import pyrealsense2 as rs

# ----------------------------
# Dynamixel 설정
# ----------------------------
ADDR_TORQUE_ENABLE          = 64
ADDR_PROFILE_VELOCITY       = 112
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4

PROTOCOL_VERSION            = 2.0
BAUDRATE                    = 1000000
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1

# ID 정의
BODY_IDS = list(range(1, 15))       # 1~14 (팔+그리퍼)
YAW_ID = 15
PITCH_ID = 16
HEAD_IDS = [YAW_ID, PITCH_ID]

HOME_POSITION = 2048
DEG_TO_TICK = 4095.0 / 360.0
MAX_ANGLE = 30.0
MAX_TICK_OFFSET = int(DEG_TO_TICK * MAX_ANGLE)

# ----------------------------
# Dynamixel 초기화
# ----------------------------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

if not portHandler.openPort():
    print("Failed to open port"); exit()
portHandler.setBaudRate(BAUDRATE)

# Torque ON + 속도 설정
for dxl_id in BODY_IDS + HEAD_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, 20)

# ----------------------------
# motion.yaml 로드
# ----------------------------
with open("motion.yaml", "r") as f:
    motion_data = yaml.safe_load(f)

home_data = motion_data["home_position"]

# 전체 Base 위치 구성 (팔+그리퍼만, 머리는 PID로 제어)
base_positions = {}
for section in ["left_arm", "right_arm", "l_gripper", "r_gripper"]:
    for k, v in home_data[section].items():
        base_positions[int(k.replace("ID",""))] = v

def move_positions(pos_dict):
    groupSyncWrite.clearParam()
    for dxl_id, goal in pos_dict.items():
        param_goal = [
            DXL_LOBYTE(DXL_LOWORD(goal)),
            DXL_HIBYTE(DXL_LOWORD(goal)),
            DXL_LOBYTE(DXL_HIWORD(goal)),
            DXL_HIBYTE(DXL_HIWORD(goal))
        ]
        groupSyncWrite.addParam(dxl_id, param_goal)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

# 몸 base 위치로 이동
move_positions(base_positions)
time.sleep(2)

# ----------------------------
# PID 컨트롤러
# ----------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.prev_error = error
        return output

yaw_pid = PID(kp=0.08, ki=0.0, kd=0.02)
pitch_pid = PID(kp=0.08, ki=0.0, kd=0.02)

yaw_pos = home_data["head"]["ID15"]
pitch_pos = home_data["head"]["ID16"]

# ----------------------------
# YOLO + Realsense
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

model = YOLO("yolov8n-face-lindevs.pt")  # 일반 YOLOv8n, person 클래스 필터링

last_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        results = model.predict(img, verbose=False)

        people = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls != 0 or conf < 0.5:  # 0=person, confidence>0.5
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                area = (x2-x1)*(y2-y1)
                people.append((cx, cy, area, x1, y1, x2, y2, conf))

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if people:
            # 가장 큰 박스 선택
            people.sort(key=lambda x:x[2], reverse=True)
            cx, cy, _, x1, y1, x2, y2, conf = people[0]

            # 화면 중심과 오차
            err_x = 320 - cx
            err_y = 240 - cy  # pitch는 반대로

            # PID 출력
            yaw_offset = yaw_pid.compute(err_x, dt)
            pitch_offset = pitch_pid.compute(err_y, dt)

            yaw_pos += int(yaw_offset)
            pitch_pos += int(pitch_offset)

            # 제한 각도 적용
            yaw_pos = max(HOME_POSITION-MAX_TICK_OFFSET, min(HOME_POSITION+MAX_TICK_OFFSET, yaw_pos))
            pitch_pos = max(HOME_POSITION-MAX_TICK_OFFSET, min(HOME_POSITION+MAX_TICK_OFFSET, pitch_pos))

            # Bounding Box & center 시각화
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, f"Person {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 사람 없으면 천천히 원점으로 복귀
            yaw_pos += int((HOME_POSITION - yaw_pos)*0.05)
            pitch_pos += int((HOME_POSITION - pitch_pos)*0.05)

        # 머리 위치 전송
        move_positions({YAW_ID: yaw_pos, PITCH_ID: pitch_pos})

        # 화면 표시
        cv2.imshow("Head Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    portHandler.closePort()
    cv2.destroyAllWindows()
