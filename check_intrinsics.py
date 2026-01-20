import pyrealsense2 as rs

def get_intrinsics():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Using the same resolution as in collect_data_lcm.py for consistency
    # D405 common resolution
    width = 848
    height = 480
    print(f"Configuring RealSense D405 with resolution {width}x{height}...")
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    try:
        # Start streaming
        profile = pipeline.start(config)
        
        # Get the stream profile for the color stream
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        print("\n=== Camera Intrinsics for D405 (Color Stream) ===")
        print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"Principal Point (ppx, ppy): ({intrinsics.ppx:.5f}, {intrinsics.ppy:.5f})")
        print(f"Focal Length (fx, fy): ({intrinsics.fx:.5f}, {intrinsics.fy:.5f})")
        print(f"Distortion Model: {intrinsics.model}")
        print(f"Distortion Coeffs: {intrinsics.coeffs}")

        # Matrix format
        print("\nIntrinsics Matrix (K):")
        print(f"[[{intrinsics.fx:.5f}, 0.00000, {intrinsics.ppx:.5f}],")
        print(f" [0.00000, {intrinsics.fy:.5f}, {intrinsics.ppy:.5f}],")
        print(f" [0.00000, 0.00000, 1.00000]]")
        
        # Get depth scale (FACTOR_DEPTH)
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            factor_depth = 1.0 / depth_scale
            print(f"\nFACTOR_DEPTH (Depth Scale): {factor_depth}")
        except Exception as e:
            print(f"\nWarning: Could not get depth scale - {e}")
        print("=")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    get_intrinsics()
