# Andrea Antonello 2024
import argparse
import logging
import cv2
import numpy as np
from datetime import datetime
import json
from config.config_manager import load_json_config, overwrite_json_config
from src.calibration import Calibration
import matplotlib.pyplot as plt
import matplotlib as mpl

def return_camera_indexes():
    index = 0
    arr = []
    i = 5
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def setup_calibration():
    print('Entering camera calibration')

def setup_source():
    print('Entering camera selection')
    sources = return_camera_indexes()
    if len(sources) != 0:
        if len(sources) == 1:
            logging.warning('Only one camera available at source 0. This input will be used.')
            return overwrite_json_config(key="source", value=int(0))
        else:
            input('\n----------------------------------------------------------------------------------------'
                  '\n-----------------------------  SOURCE  SELECTION  UTILITY  -----------------------------'
                  '\n----------------------------------------------------------------------------------------\n'
                  'Press enter to visualise the available camera sources. '
                  'Please note down the source number you desire. '
                  'Press ESC to exit the imshow() \n')
            for source in sources:
                cap = cv2.VideoCapture(source)
                ret, frame = cap.read()
                cv2.putText(frame, str(source), (int(frame.shape[1]/2), int(frame.shape[0]/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 20)
                cv2.imshow('Source ' + str(source), frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            input_source = input(f'Please enter the camera source you desire to use. Available sources are {sources}\n')
            if int(input_source) in sources:
                return overwrite_json_config(key="source", value=int(input_source))
            else:
                raise ValueError('Selected input source is not available')
    else:
        raise ValueError('No input sources detected')

def detect_aruco_and_estimate_pose():
    source = int(load_json_config()['source'])
    cap = cv2.VideoCapture(source)

    # Load camera parameters from configuration
    config = load_json_config()
    camera_matrix = np.array(load_json_config()['mtx'])
    dist_coeffs = np.array(config['dist'])

    # Define ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
    aruco_params = cv2.aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error('Failed to grab frame from camera')
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                # Draw axis for the marker
                # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                thickness = 5
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03, thickness)

                # Display translation and rotation vectors
                text = f"Tvec: {tvec.flatten()} Rvec: {rvec.flatten()}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Marker detected at Tvec: {tvec.flatten()} Rvec: {rvec.flatten()}")

        # Display the frame
        cv2.imshow('Aruco Marker Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def generate_aruco_markers_pdf():
    print('Generating sample ArUco markers PDF')
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)

    fig = plt.figure()
    nx = 4
    ny = 3
    for i in range(1, nx * ny + 1):
        ax = fig.add_subplot(ny, nx, i)
        img = cv2.aruco.generateImageMarker(aruco_dict, i, 700)
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis('off')

    pdf_filename = "aruco_markers.pdf"
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Sample ArUco markers saved to {pdf_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description='Aruco Tags Examples',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--calibrate',
        action='store_true', default=False, help='Performs the intrinsic camera calibration'
    )

    parser.add_argument(
        '--source', '-r',
        action='store_true', default=False, help='Selects camera source'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true', help='Increases the output verbosity')

    parser.add_argument(
        '--generate_markers', '-g',
        action='store_true', default=False, help='Generate a PDF of sample ArUco markers'
    )

    parser.add_argument(
        '--aruco', '-a',
        action='store_true', help='Detects Aruco markers and performs pose estimation')

    return parser.parse_args()

if __name__ == '__main__':
    aruco_type = cv2.aruco.DICT_6X6_250

    while int(load_json_config()['source']) == -1:
        setup_source()

    ns = parse_args()

    if ns.calibrate:
        while int(load_json_config()['source']) == -1:
            setup_source()
        cal = Calibration(load_json_config())
        cal.calibrate_from_images()

    elif ns.source:
        setup_source()
        while int(load_json_config()['source']) == -1:
            setup_source()

    elif ns.aruco:
        detect_aruco_and_estimate_pose()

    elif ns.generate_markers:
        generate_aruco_markers_pdf()

    else:
        logging.error('Please specify program mode.')
        raise SystemExit
