import numpy as np
import cv2
import os
import glob
import time
from config.config_manager import load_json_config, overwrite_json_config


class Calibration:
    def __init__(self, config, images_path='config/temp/'):
        self.source = config["source"]
        self.images_path = images_path

    def get_images(self, max_count=30):
        images = glob.glob(self.images_path + '*')

        for f in images:
            os.remove(f)

        t0 = time.time()
        snap_delta = 5  # take a calibration snap every snap_delta [s]
        count = 0
        while True and count < max_count:
            t1 = time.time()
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            ret, frame = cap.read()
            frame_clone = frame.copy()
            current_delta = int(t1 - t0)
            next_snap = snap_delta - current_delta
            text = f'next snap in {next_snap} s'
            cv2.putText(frame_clone, text, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

            # Display the resulting frame
            cv2.imshow('preview', frame_clone)
            if next_snap == 0:
                t0 = time.time()
                cv2.imwrite(self.images_path + 'frame%d.jpg' % count, frame)
                cv2.putText(frame_clone, 'SNAP!', (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
                cv2.imshow('preview', frame_clone)
                cv2.waitKey(1000)
                count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def calibrate_from_images(self):
        images = glob.glob(self.images_path + '*')
        print('Searching for existing calibration images...')
        if images:
            answer = ""
            while answer not in ['Y', 'N']:
                answer = input('The config/temp already contains calibration images. '
                               'Do you want to overwrite them? (''Y/N) \n'
                               '(This will open the get_images GUI) \n')
            if answer == 'Y':
                print('Entering get_images GUI...')
                self.get_images()
        else:
            print('Calibration images not found. Entering get_images GUI...')
            self.get_images()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        chessboard = [9, 6]
        objp = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        images = glob.glob(self.images_path + '*')

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard[0], chessboard[1]), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (chessboard[0], chessboard[1]), corners2, ret)
                cv2.imshow('Calibration input', img)
                cv2.waitKey(50)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("ret", ret)
        print("mtx", mtx)
        print("dist", dist)

        for image in images:
            img = cv2.imread(image)
            h, w = img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # Un-distort
            dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
            # Crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv2.imshow('Calibration output', dst)
            cv2.waitKey(50)

        mean_error = 0
        for point in range(len(objpoints)):
            im_points, _ = cv2.projectPoints(objpoints[point], rvecs[point], tvecs[point], mtx, dist)
            error = cv2.norm(imgpoints[point], im_points, cv2.NORM_L2)/len(im_points)
            mean_error += error

        if ret > 1:
            raise ValueError('Calibration error is too large. Please retake calibration images.\n'
                             'Make sure the board has been printed correctly, it is on a flat surface and there are'
                             'no creases/bends.')
        print("Total error: {}".format(mean_error/len(objpoints)))
        overwrite_json_config(key="ret", value=ret)
        overwrite_json_config(key="mtx", value=mtx.tolist())
        overwrite_json_config(key="dist", value=dist.tolist())
