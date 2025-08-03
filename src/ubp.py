 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tty
import termios
import yaml
from dynamixel_sdk import *

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

MY_DXL = 'MX_SERIES'

# Control table address
ADDR_TORQUE_ENABLE          = 64
ADDR_PROFILE_VELOCITY       = 112
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_POSITION       = 132

LEN_GOAL_POSITION           = 4
LEN_PRESENT_POSITION        = 4

DXL_MINIMUM_POSITION_VALUE  = 0
DXL_MAXIMUM_POSITION_VALUE  = 4095
HOME_POSITION               = 2048  # 초기 원점
PROFILE_VELOCITY            = 10
BAUDRATE                    = 1000000
PROTOCOL_VERSION            = 2.0
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1
TORQUE_DISABLE              = 0
DXL_MOVING_STATUS_THRESHOLD = 20

DXL_IDS = list(range(1, 13))  # 1~12

# YAML 불러오기
with open("motion.yaml", "r") as f:
    motion_data = yaml.safe_load(f)

# YAML에서 base 위치 추출
base_positions = {}
base_positions.update(motion_data["home_position"]["left_arm"])
base_positions.update(motion_data["home_position"]["right_arm"])

# Open port
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

if not portHandler.openPort():
    print("Failed to open the port")
    quit()
print("Succeeded to open the port")

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    quit()
print("Succeeded to set baudrate")

# Torque ON + Profile velocity 설정
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    print(f"[ID:{dxl_id}] Torque {'enabled' if dxl_error == 0 else 'error'}")
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
    groupSyncRead.addParam(dxl_id)

# --------------------
# 1단계: 원점 이동
# --------------------
def move_to_position(goal_positions_dict):
    groupSyncWrite.clearParam()
    for dxl_id in DXL_IDS:
        goal = goal_positions_dict[dxl_id] if isinstance(goal_positions_dict, dict) else goal_positions_dict
        param_goal = [
            DXL_LOBYTE(DXL_LOWORD(goal)),
            DXL_HIBYTE(DXL_LOWORD(goal)),
            DXL_LOBYTE(DXL_HIWORD(goal)),
            DXL_HIBYTE(DXL_HIWORD(goal))
        ]
        groupSyncWrite.addParam(dxl_id, param_goal)
    dxl_comm_result = groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    # 목표 도달 대기
    while True:
        groupSyncRead.txRxPacket()
        all_reached = True
        for dxl_id in DXL_IDS:
            present_position = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            goal = goal_positions_dict[dxl_id] if isinstance(goal_positions_dict, dict) else goal_positions_dict
            diff = abs(goal - present_position)
            print(f"[ID:{dxl_id}] Goal: {goal}  Present: {present_position}  Δ={diff}")
            if diff > DXL_MOVING_STATUS_THRESHOLD:
                all_reached = False
        if all_reached:
            break

# 1) 원점 이동
print("\nMoving to HOME_POSITION (2048)...")
move_to_position(HOME_POSITION)

# 2) Base 위치 이동
print("\nMoving to BASE position from motion.yaml...")
move_to_position({int(k.replace("ID","")):v for k,v in base_positions.items()})

print("Motion complete. Press any key to exit...")
getch()

portHandler.closePort()
