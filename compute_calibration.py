# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os

import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

iamges_path = "./data_old1"  # 手眼标定采集的标定版图片所在路径
arm_pose_file = "./data_old1/poses.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


def euler_angles_to_rotation_matrix(rx, ry, rz):
    """
    Convert roll-pitch-yaw angles to rotation matrix
    """
    roll, pitch, yaw = rx, ry, rz
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


def camera_calibrate(iamges_path):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板的中长度对应的角点的个数
    YY = 8  # 标定板的中宽度对应的角点的个数
    L = 0.02  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    valid_indices = [] # 存储有效图片的索引

    for i in range(0, 30):  # 标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一般采集20张左右就够，实际情况可修改

        image = f"{iamges_path}/{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):

            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            # h_init, width_init = img.shape[:2]
            # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
            # print(f"图像大小(resize)： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            # 只有当角点检测成功才加入计算列表
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:
                # print(corners)
                print(f"左上角点：{corners[0, 0]}")
                print(f"右下角点：{corners[-1, -1]}")

                # 绘制角点并显示图像
                cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
                cv2.imshow('Chessboard', img)

                cv2.waitKey(3000)  ## 停留1s, 观察找到的角点是否正确

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if corners2 is not None:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
                
                # 记录成功检测到角点的图片索引，用于后续筛选机械臂位姿
                valid_indices.append(i)
            else:
                print(f"警告：无法在图片 {image} 中检测到角点，已跳过该张图片。")

    N = len(img_points)
    print(f"成功处理图片数量：{N}")

    # 标定得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs, valid_indices


def process_arm_pose(arm_pose_file, valid_indices):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    
    # 只处理有效图片对应的位姿
    for idx in valid_indices:
        if idx < len(all_lines):
            line = all_lines[idx]
            pose = [float(v) for v in line.split(',')]
            R, t = pose_to_homogeneous_matrix(pose=pose)
            R_arm.append(R)
            t_arm.append(t)
        else:
            print(f"警告: 索引 {idx} 超出了位姿文件的行数范围，该数据点将被忽略。")
            
    return R_arm, t_arm


def invert_pose(R, t):
    """
    Calculate the inverse of a rigid body transformation.
    Given R, t such that P_new = R * P_old + t
    Returns R_inv, t_inv such that P_old = R_inv * P_new + t_inv
    """
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    return R_inv, t_inv


def hand_eye_calibrate():
    rvecs, tvecs, valid_indices = camera_calibrate(iamges_path=iamges_path)
    # 将 valid_indices 传递给 process_arm_pose，确保位姿与图片一一对应
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file, valid_indices=valid_indices)
    
    # 确保用于手眼标定的数据长度一致
    assert len(R_arm) == len(rvecs) == len(tvecs), "数据长度不一致！"

    print("\n========== 1. cv2.calibrateHandEye (5 Methods) ==========")
    # Methods for cv2.calibrateHandEye
    methods_he = [
        ("Tsai", cv2.CALIB_HAND_EYE_TSAI),
        ("Park", cv2.CALIB_HAND_EYE_PARK),
        ("Horaud", cv2.CALIB_HAND_EYE_HORAUD),
        ("Andreff", cv2.CALIB_HAND_EYE_ANDREFF),
        ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS)
    ]

    # R_arm is R_gripper2base (Gripper -> Base) because it's the pose of gripper in base.
    # calibHandEye expects gripper2base and target2cam (which is rvecs/tvecs)
    
    for name, method in methods_he:
        try:
            R_cam2grp, t_cam2grp = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, method=method)
            print(f"--- Method: {name} ---")
            print("R_cam2gripper:\n", R_cam2grp)
            print("t_cam2gripper:\n", t_cam2grp.flatten())
        except Exception as e:
            print(f"--- Method: {name} Failed: {e} ---")

    print("\n========== 2. cv2.calibrateRobotWorldHandEye (2 Methods) ==========")
    # Methods for cv2.calibrateRobotWorldHandEye
    methods_rwhe = [
        ("Shah", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
        ("Li", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
    ]

    # calibrateRobotWorldHandEye expects:
    # 1. R_world2cam, t_world2cam (World -> Camera). this is rvecs, tvecs.
    # 2. R_base2gripper, t_base2gripper (Base -> Gripper).
    #    We have R_arm (Gripper -> Base). So we need to INVERT it.
    
    R_b2g_list, t_b2g_list = [], []
    for R, t in zip(R_arm, t_arm):
        Ri, ti = invert_pose(R, t)
        R_b2g_list.append(Ri)
        t_b2g_list.append(ti)
    
    for name, method in methods_rwhe:
        try:
            R_base2world, t_base2world, R_grp2cam, t_grp2cam = cv2.calibrateRobotWorldHandEye(
                rvecs, tvecs, R_b2g_list, t_b2g_list, method=method
            )
            
            # Result is Gripper -> Camera. We want Camera -> Gripper.
            # So invert the result.
            R_cam2grp, t_cam2grp = invert_pose(R_grp2cam, t_grp2cam)
            
            print(f"--- Method: {name} ---")
            print("R_cam2gripper:\n", R_cam2grp)
            print("t_cam2gripper:\n", t_cam2grp.flatten())
            # print("R_base2world:\n", R_base2world) # Optional: World wrt Base
        except Exception as e:
            print(f"--- Method: {name} Failed: {e} ---")

    print("+++++++++++++++++++++++++++++++++++++++++++")

    # Validation: Check Board consistency in Base frame using the last computed calibrationResult
    # T_board_in_base = T_end_base * T_cam_end * T_board_cam
    print("\n=== Validation (Board Position in Base) ===")
    
    RT_cam2end = np.eye(4)
    RT_cam2end[:3, :3] = R_cam2grp
    RT_cam2end[:3, 3] = t_cam2grp.flatten()
    
    for k in range(min(10, len(R_arm))): 
        RT_eb = np.eye(4)
        RT_eb[:3, :3] = R_arm[k]
        RT_eb[:3, 3] = t_arm[k].flatten()
        
        RT_bc = np.eye(4)
        rmat, _ = cv2.Rodrigues(rvecs[k])
        RT_bc[:3, :3] = rmat
        RT_bc[:3, 3] = tvecs[k].flatten()
        
        RT_final = RT_eb @ RT_cam2end @ RT_bc
        print(f"Frame {k} pos: {RT_final[:3, 3].flatten()}")

    # Return last result or one of them (e.g. Shah) for main block compatibility
    return R_cam2grp, t_cam2grp


if __name__ == "__main__":
    R, t = hand_eye_calibrate()

    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)
