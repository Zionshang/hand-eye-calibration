import time
import json
import cv2
import click
import numpy as np
import pyrealsense2 as rs
from pynput import keyboard
from communication.lcm.lcm_client import Arx5LcmClient

# --- Global Vars (Loaded in main) ---
CAMERA_MATRIX = None
DIST_COEFFS = None
H_HAND_EYE = None

XX, YY = 11, 8
SQUARE_SIZE = 0.02

def load_calibration_data(method_name="Tsai"):
    global CAMERA_MATRIX, DIST_COEFFS, H_HAND_EYE
    with open("calibration_result.json", "r") as f:
        data = json.load(f)
    
    CAMERA_MATRIX = np.array(data["intrinsics"]["camera_matrix"])
    DIST_COEFFS = np.array(data["intrinsics"]["dist_coeffs"])
    
    if method_name not in data["methods"]:
        print(f"Warning: Method {method_name} not found. Using Tsai.")
        method_name = "Tsai"
        
    m = data["methods"][method_name]
    R = np.array(m["R"])
    t = np.array(m["T"]).reshape(3, 1)
    
    H_HAND_EYE = np.eye(4)
    H_HAND_EYE[:3, :3] = R
    H_HAND_EYE[:3, 3] = t.flatten()
    print(f"Loaded Calibration: {method_name}")


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
    
    success, rvec, tvec = cv2.solvePnP(objp, corners2, CAMERA_MATRIX, DIST_COEFFS)
    if not success: return None

    # Target the 1st Row, 1st Column corner (Top-Left) (0-based index: Row 0, Col 0)
    # Coordinate in Object Frame: (0, 0, 0)
    target_obj_point = np.array([[0.0, 0.0, 0.0]]) 

    # Visualize target point (Project target_obj_point to image)
    points_2d, _ = cv2.projectPoints(target_obj_point, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    target_pix = tuple(map(int, points_2d[0].ravel()))

    cv2.drawChessboardCorners(image, (XX, YY), corners2, ret)
    cv2.circle(image, target_pix, 10, (0, 0, 255), -1)
    cv2.imshow("Corners", image)
    cv2.waitKey(1)

    # Transform target point from Object Frame to Camera Frame
    # P_cam = R * P_obj + T
    R_mat, _ = cv2.Rodrigues(rvec)
    p_in_cam_3d = R_mat @ target_obj_point.T + tvec # shape (3, 1)
    p_in_cam = np.append(p_in_cam_3d.flatten(), 1.0)

    # Result = T_base_end @ T_end_cam @ P_in_cam
    p_in_base = pose_to_H(current_ee_pose) @ H_HAND_EYE @ p_in_cam
    
    target_cmd = np.zeros(6)
    target_cmd[:3] = p_in_base[:3]
    target_cmd[3:] = [0.0, np.pi / 2.0, 0.0] # Pitch 90 deg
    target_cmd[2] += 0.03  # Offset above the board
    return target_cmd

def start_verification_task(client, pipeline):
    print("\n=== Verification Started ===")
    print(" [H]     : Touch Target")
    print(" [Space] : Reset Home")
    print(" [Q]     : Quit")

    key_map = {
        'h': keyboard.KeyCode.from_char("h"),
        'q': keyboard.KeyCode.from_char("q"),
        'space': keyboard.Key.space
    }
    # Flags to latch events
    # We use a dict to store the state. True = Pending Request.
    key_requests = {k: False for k in key_map.values()}

    def on_press(key):
        if key in key_requests: 
            key_requests[key] = True

    # Remove on_release reset to avoid missing fast taps
    # The flag will be reset manually in the main loop after processing
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    # Checking Camera Stream & Warping Up
    for i in range(30):
        frames = pipeline.poll_for_frames()
        if frames:
            color = frames.get_color_frame()
            if color:
                print("[Init] Camera OK!")
                break
        time.sleep(0.1)
    else:
        print("[Warning] Camera stream check timeout!")

    # Init robot pose with continuous command stream
    target_ready = np.array([0.25, 0.0, 0.17, 0.0, 1.2, 0.0])

    # Send command continuously to ensure execution 
    client.set_ee_pose(target_ready, gripper_pos=0.0, preview_time=3.0)
    time.sleep(3.5)
    
    start_time = time.monotonic()
    
    # We remove last_h logic because we now consume events
    # last_h = False 

    while True:
        # 1. Update State & Print
        # ... existing code ...
        state = client.get_state()
        ee_pose = state["ee_pose"]
        print(f"Time: {time.monotonic() - start_time:.1f}s | Pose: {ee_pose[:3]}", end="\r")

        # Handle cv2 keys (backup for window focus)
        # Using waitKey(1) inside loop is good for GUI responsiveness
        # And we merge it into our request flags
        key_cv = cv2.waitKey(1) & 0xFF
        if key_cv == ord('h'): key_requests[key_map['h']] = True
        elif key_cv == ord('q'): key_requests[key_map['q']] = True
        elif key_cv == 32: key_requests[key_map['space']] = True # Space

        # 2. Process Requests
        if key_requests[key_map['q']]: 
            break
        
        if key_requests[key_map['space']]:
            print("\n[Cmd] Resetting to Home...", end="\r")
            client.reset_to_home()
            start_time = time.monotonic()
            key_requests[key_map['space']] = False # Reset flag
            continue

        frames = pipeline.poll_for_frames()
        img = None
        if frames:
            color = frames.get_color_frame()
            if color:
                img = np.asanyarray(color.get_data())
                cv2.imshow("Preview", img)
                # waitKey is already called above

        if key_requests[key_map['h']] and img is not None:
            print("\n[Cmd] Calculating Target...")
            target = get_target_touch_pose(img, ee_pose)
            if target is not None:
                print(f" -> Moving to: {target}")
                client.set_ee_pose(target, gripper_pos=0, preview_time=3.0)
            else:
                print(" -> Board not found!")
            
            # Reset flag after processing (Debounce/One-shot)
            key_requests[key_map['h']] = False 

        # Short sleep to prevent CPU hogging (optional if IO bound)
        time.sleep(0.01)

@click.command()
@click.option("--address", default="239.255.76.67", help="LCM address")
@click.option("--port", default=7667, help="LCM port")
@click.option("--method", default="Tsai", help="Calib method: Tsai, Park, Horaud, Daniilidis")
def main(address, port, method):
    load_calibration_data(method)

    client = Arx5LcmClient(url="", address=address, port=port, ttl=1)
    client.reset_to_home()
    print("Robot initialized.")

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
