import time
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from dynamixel_sdk import *
import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment  # 헝가리안 알고리즘

# ----------------------------
# Dynamixel & PID 설정
# ----------------------------
ADDR_TORQUE_ENABLE          = 64
ADDR_PROFILE_VELOCITY       = 112
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4

PROTOCOL_VERSION            = 2.0
BAUDRATE                    = 1000000
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1

BODY_IDS = list(range(1, 15))   # 팔+그리퍼
YAW_ID = 15
PITCH_ID = 16
HEAD_IDS = [YAW_ID, PITCH_ID]

HOME_POSITION = 2048
DEG_TO_TICK = 4095.0 / 360.0
MAX_ANGLE = 30.0
MAX_TICK_OFFSET = int(DEG_TO_TICK * MAX_ANGLE)

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




# ----------------------------
# 간단한 SORT 구현
# ----------------------------
class Track:
    count = 0

    def __init__(self, bbox):
        self.id = Track.count
        Track.count += 1
        self.bbox = bbox
        self.hits = 0
        self.no_losses = 0

        # Kalman Filter (x, y, s, r, dx, dy, ds)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1,0,0,0,dt,0,0],
            [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.P *= 10.
        self.kf.R *= 0.01
        self.kf.Q *= 0.01

        self.update(bbox)

    def predict(self):
        self.kf.predict()
        x, y, s, r = self.kf.x[:4]

        # NaN 방지
        s = max(float(s), 1e-6)
        r = max(float(r), 1e-6)

        w = np.sqrt(s * r)
        h = s / w
        self.bbox = [x-w/2, y-h/2, x+w/2, y+h/2]
        return self.bbox


    def update(self, bbox):
        x1,y1,x2,y2 = bbox[:4]
        w, h = x2-x1, y2-y1
        x, y = x1+w/2, y1+h/2
        s = w*h
        r = w/float(h)
        self.kf.update([x, y, s, r])
        self.bbox = bbox
        self.hits += 1

class SimpleSORT:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2-xx1)
        h = np.maximum(0., yy2-yy1)
        wh = w*h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
                  (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return o

    def associate(self, dets, trks):
        if len(trks)==0:
            return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0),dtype=int)

        iou_matrix = np.zeros((len(dets),len(trks)),dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d,t] = self.iou(det, trk)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        for r,c in zip(row_ind, col_ind):
            if iou_matrix[r,c] < self.iou_threshold:
                continue
            matches.append([r,c])
        matches = np.array(matches)

        if matches.size > 0:
            unmatched_dets = [d for d in range(len(dets)) if d not in matches[:,0]]
            unmatched_trks = [t for t in range(len(trks)) if t not in matches[:,1]]
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(trks)))

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)


    def update(self, detections):
        predictions = []
        for t in self.tracks:
            predictions.append(t.predict())

        matches, unmatched_dets, unmatched_trks = self.associate(detections, predictions)

        for t_idx in range(len(self.tracks)):
            if t_idx in unmatched_trks:
                self.tracks[t_idx].no_losses += 1
            else:
                d_idx = matches[np.where(matches[:,1]==t_idx)[0],0]
                self.tracks[t_idx].update(detections[d_idx[0]])

        for i in unmatched_dets:
            self.tracks.append(Track(detections[i]))

        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]

        results = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                x1,y1,x2,y2 = t.bbox[:4]
                results.append([x1,y1,x2,y2,t.id])
        return results

# ----------------------------
# 초기화 함수
# ----------------------------
def initialize_dynamixel():
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    if not portHandler.openPort():
        print("Failed to open port"); exit()
    portHandler.setBaudRate(BAUDRATE)

    for dxl_id in BODY_IDS + HEAD_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, 20)

    return portHandler, packetHandler, groupSyncWrite

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def initialize_yolo(model_path="yolov8n-face-lindevs.pt"):
    return YOLO(model_path)

def initialize_sort_tracker():
    return SimpleSORT(max_age=10, min_hits=3, iou_threshold=0.3)

def initialize_ekf():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1/30.0
    kf.F = np.array([[1,0,dt,0],
                     [0,1,0,dt],
                     [0,0,1,0],
                     [0,0,0,1]])
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]])
    kf.P *= 1000.0
    kf.R *= 5.0
    kf.Q *= 0.01
    return kf


# ----------------------------
# 보조 함수
# ----------------------------
def move_positions(groupSyncWrite, pos_dict):
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

def detect_people(model, img):
    results = model.predict(img, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls == 0 and conf > 0.5:  # 사람만
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])
    return np.array(detections)

def select_and_filter_target(tracks, kf):
    if len(tracks) == 0:
        return None

    # 가장 큰 박스 선택
    tracks = sorted(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
    x1, y1, x2, y2, track_id = tracks[0]
    cx, cy = (x1+x2)/2, (y1+y2)/2

    # EKF 필터링
    kf.predict()
    kf.update(np.array([cx, cy]))
    filtered_cx = kf.x[0].item()
    filtered_cy = kf.x[1].item()

    # NaN 방어
    if np.isnan(filtered_cx) or np.isnan(filtered_cy):
        return None

    return int(filtered_cx), int(filtered_cy)


# ----------------------------
# 메인 루프
# ----------------------------
def main():
    # 1. 초기화
    portHandler, packetHandler, groupSyncWrite = initialize_dynamixel()
    pipeline = initialize_realsense()
    model = initialize_yolo()
    tracker = initialize_sort_tracker()
    kf = initialize_ekf()

    # 2. motion.yaml 로드 후 홈포지션 이동
    with open("motion.yaml", "r") as f:
        motion_data = yaml.safe_load(f)
    home_data = motion_data["home_position"]

    # 팔 + 그리퍼 기본 위치
    base_positions = {}
    for section in ["left_arm", "right_arm", "l_gripper", "r_gripper"]:
        for k, v in home_data[section].items():
            base_positions[int(k.replace("ID",""))] = v

    move_positions(groupSyncWrite, base_positions)
    time.sleep(2)

    # PID 초기화
    yaw_pid = PID(0.08, 0, 0.02)
    pitch_pid = PID(0.08, 0, 0.02)
    yaw_pos = home_data["head"]["ID15"]
    pitch_pos = home_data["head"]["ID16"]

    last_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            # YOLO 탐지 + SORT 추적
            detections = detect_people(model, img)
            tracks = tracker.update(detections)

            # 목표 좌표 (EKF 필터링)
            target = select_and_filter_target(tracks, kf)
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if target:
                cx, cy = target
                err_x = 320 - cx
                err_y = 240 - cy
                yaw_pos += int(yaw_pid.compute(err_x, dt))
                pitch_pos += int(pitch_pid.compute(err_y, dt))
            else:
                # 목표 없으면 원점으로 복귀
                yaw_pos += int((HOME_POSITION - yaw_pos)*0.05)
                pitch_pos += int((HOME_POSITION - pitch_pos)*0.05)

            # 범위 제한
            yaw_pos = max(HOME_POSITION-MAX_TICK_OFFSET, min(HOME_POSITION+MAX_TICK_OFFSET, yaw_pos))
            pitch_pos = max(HOME_POSITION-MAX_TICK_OFFSET, min(HOME_POSITION+MAX_TICK_OFFSET, pitch_pos))

            # Dynamixel 명령 전송
            move_positions(groupSyncWrite, {YAW_ID: yaw_pos, PITCH_ID: pitch_pos})

            # 화면 표시
            cv2.imshow("Head Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        portHandler.closePort()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
