import sys
sys.path.append('/home/pi/TurboPi/')
import time
import signal
import HiwonderSDK.mecanum as mecanum
from Camera2 import Camera, capture_images  # Import from Camera2.py
import cv2  # Import cv2

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# Initialize chassis and camera
chassis = mecanum.MecanumChassis()
camera = Camera()

start = True

def Stop(signum, frame):
    global start
    start = False
    print('Shutting down...')
    chassis.set_velocity(0, 0, 0)
    camera.camera_close()
    cv2.destroyAllWindows()

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    try:
        print("Starting movement...")
        chassis.set_velocity(50, 90, 0)
        time.sleep(3)
        chassis.set_velocity(0, 0, 0)
        print("Opening camera...")
        camera.camera_open()
        time.sleep(2)  # Allow the camera to initialize

        # Enter image capture mode
        print("Camera ready for capturing images. Press 'c' to capture and 'ESC' to exit.")
        capture_images(camera)  # Function to handle capturing images and exiting

    finally:
        # Ensure proper cleanup
        chassis.set_velocity(0, 0, 0)
        camera.camera_close()
        cv2.destroyAllWindows()
        print("Finished")
