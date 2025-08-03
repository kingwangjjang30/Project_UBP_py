#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tty
import termios
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

# ------------------ 설정 ------------------
ADDR_TORQUE_ENABLE    = 64
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION  = 4

TORQUE_ENABLE         = 1
TORQUE_DISABLE        = 0
PROTOCOL_VERSION      = 2.0
DEVICENAME            = '/dev/ttyUSB0'
BAUDRATE              = 1000000

DXL_IDS = list(range(1, 13))  # 1~12
# ------------------------------------------

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

# 1. 포트 연결
if not portHandler.openPort():
    print("Failed to open the port")
    quit()
print("Succeeded to open the port")

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    quit()
print("Succeeded to set baudrate")

# 2. 모든 모터 토크 OFF, GroupSyncRead 등록
for dxl_id in DXL_IDS:
    # Torque OFF
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Torque OFF")
    else:
        print(f"[ID:{dxl_id}] {packetHandler.getTxRxResult(dxl_comm_result)}")

    # GroupSyncRead 등록
    if not groupSyncRead.addParam(dxl_id):
        print(f"[ID:{dxl_id}] groupSyncRead addparam failed")
        quit()

print("\n--- Ready ---")
print("Press 't' to enable torque and read positions. Press 'q' to quit.")

# 3. 키 입력 대기
while True:
    key = getch()
    if key == 'q':
        break
    elif key == 't':
        print("\nEnabling torque and reading positions...\n")

        # 토크 ON
        for dxl_id in DXL_IDS:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result == COMM_SUCCESS:
                print(f"[ID:{dxl_id}] Torque ON")
            else:
                print(f"[ID:{dxl_id}] {packetHandler.getTxRxResult(dxl_comm_result)}")

        # 현재 위치 읽기
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"SyncRead Tx Error: {packetHandler.getTxRxResult(dxl_comm_result)}")

        for dxl_id in DXL_IDS:
            if groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                present_position = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                print(f"[ID:{dxl_id}] Present Position: {present_position}")
            else:
                print(f"[ID:{dxl_id}] Failed to get Present Position")

print("Closing port...")
portHandler.closePort()
print("Done.")
