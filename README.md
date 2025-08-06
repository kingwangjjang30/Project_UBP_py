# Installation & Setup

## 1️⃣ Requirements

### System Packages
```bash

sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    libusb-1.0-0-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    libgtk-3-dev

```


<<<<<<< HEAD
### Python Environment
```bash

python3 -m venv headtrack_env
source headtrack_env/bin/activate
pip install --upgrade pip

```


=======
>>>>>>> origin/develop
### Python Packages
```bash

# Dynamixel SDK
pip install dynamixel-sdk

# YOLOv8
pip install ultralytics

# OpenCV
pip install opencv-python

# Intel Realsense Python binding
pip install pyrealsense2

# YAML & NumPy
pip install pyyaml numpy

```


## 2️⃣ YOLOv8-Face Model

- Download Link: ["YOLO_face".pt_link](https://github.com/lindevs/yolov8-face)

- Place the model in the project src/ folder:
```bash

yolov8n-face-lindevs.pt

```


##  3️⃣ Run

```bash

cd ~/ubp_py/src
python3 ubp_headtrack.py
Press q to quit

```


## 4️⃣ Base Motion (motion.yaml)
```yaml

home_position:
  head:
    ID15: 2048
    ID16: 2048
  left_arm:
    ID1: 2300
    ID3: 2050
    ID5: 2030
    ID7: 1390
    ID9: 2090
    ID11: 2520
  right_arm:
    ID2: 1795
    ID4: 2045
    ID6: 2065
    ID8: 2705
    ID10: 2005
    ID12: 1575
  l_gripper:
    ID13: 2048
  r_gripper:
    ID14: 2048

```
<<<<<<< HEAD

=======
>>>>>>> origin/develop
