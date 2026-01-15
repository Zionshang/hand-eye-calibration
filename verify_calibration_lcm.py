import time
import cv2
import click
import numpy as np
import pyrealsense2 as rs
from pynput import keyboard
from communication.lcm.lcm_client import Arx5LcmClient

# --- Configuration & Calibration Constants ---
CAMERA_MATRIX = np.array([
    [435.75595576, 0.0, 423.51395880],
    [0.0, 435.67409149, 243.52290173],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([-0.06143892, 0.11244468, -0.00089222, 0.00102268, -0.09769306])

R_HAND_EYE = np.array([
    [-0.01029582, -0.29120562, 0.95660508],
    [-0.99972631, -0.01710000, -0.01596544],
    [0.02100717, -0.95650765, -0.29094986]
])
T_HAND_EYE = np.array([[-0.13248165], [0.01369959], [0.09316643]])

H_HAND_EYE = np.eye(4)
H_HAND_EYE[:3, :3] = R_HAND_EYE
H_HAND_EYE[:3, 3] = T_HAND_EYE.flatten()

XX, YY = 11, 8
SQUARE_SIZE = 0.015

def euler_to_R(rx, ry, rz):
    roll, pitch, yaw = rx, ry, rz
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def pose_to_H(pose):
    T = np.eye(4)
    T[:3, :3] = euler_to_R(*pose[3:])
    T[:3, 3] = pose[:3]
    return T

def get_target_touch_pose(image, current_ee_pose):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
    if not ret: return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * SQUARE_SIZE
    
    success, _, tvec = cv2.solvePnP(objp, corners2, CAMERA_MATRIX, DIST_COEFFS)
    if not success: return None
    p_in_cam = np.append(tvec.flatten(), 1.0)

    # Result = T_base_end @ T_end_cam @ P_in_cam
    p_in_base = pose_to_H(current_ee_pose) @ H_HAND_EYE @ p_in_cam
    
    target_cmd = np.zeros(6)
    target_cmd[:3] = p_in_base[:3]
    target_cmd[3:] = [0.0, np.pi / 2.0, 0.0] # Pitch 90 deg
    return target_cmd

def start_verification_task(client, pipeline):
    print("\n=== Verification Started ===")
    print(" [H]     : Touch Target")
    print(" [Space] : Reset Home")
    print(" [Q]     : Quit")

    key_pressed = {
        keyboard.KeyCode.from_char("h"): False,
        keyboard.KeyCode.from_char("q"): False,
        keyboard.Key.space: False
    }

    def on_press(key): 
        if key in key_pressed: key_pressed[key] = True
    def on_release(key): 
        if key in key_pressed: key_pressed[key] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    
    # Checking Camera Stream
    for _ in range(30):
        if pipeline.poll_for_frames(): break
    else:
        print("[Error] No Frames Received!")
        return

    # Moving to Ready Pose
    client.set_ee_pose(np.array([0.16, 0.0, 0.3, 0.0, 0.62, 0.0]), gripper_pos=0, preview_time=3.0)
    time.sleep(3.0)

    start_time = time.monotonic()
    last_h = False

    while True:
        # 1. Update State & Print
        state = client.get_state()
        ee_pose = state["ee_pose"]
        print(f"Time: {time.monotonic() - start_time:.1f}s | Pose: {ee_pose[:3]}", end="\r")

        # 2. Key Inputs
        key_h = key_pressed[keyboard.KeyCode.from_char("h")]
        key_q = key_pressed[keyboard.KeyCode.from_char("q")]
        key_space = key_pressed[keyboard.Key.space]

        if key_q: break
        
        if key_space:
            print("\n[Cmd] Resetting to Home...", end="\r")
            client.reset_to_home()
            start_time = time.monotonic()
            continue

        frames = pipeline.poll_for_frames()
        img = None
        if frames:
            color = frames.get_color_frame()
            if color:
                img = np.asanyarray(color.get_data())
                cv2.imshow("Preview", img)
                cv2.waitKey(1)

        if key_h and not last_h and img is not None:
            print("\n[Cmd] Calculating Target...")
            target = get_target_touch_pose(img, ee_pose)
            if target is not None:
                print(f" -> Moving to: {target}")
                client.set_ee_pose(target, gripper_pos=0, preview_time=3.0)
            else:
                print(" -> Board not found!")

        last_h = key_h
        time.sleep(0.01)

@click.command()
@click.option("--address", default="239.255.76.67", help="LCM address")
@click.option("--port", default=7667, help="LCM port")
def main(address, port):
    client = Arx5LcmClient(url="", address=address, port=port, ttl=1)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        start_verification_task(client, pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        pipeline.stop()
        cv2.destroyAllWindows()
        client.reset_to_home()

if __name__ == "__main__":
    main()
