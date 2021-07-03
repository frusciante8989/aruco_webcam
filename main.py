import argparse
import logging
from datetime import datetime
import json
from config.config_manager import load_json_config, overwrite_json_config
from src.calibration import Calibration
import cv2


def return_camera_indexes():
    # Checks the first 10 indexes.
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
    #
    # parser.add_argument(
    #     '--save', '-s',
    #     action='store_true', default=False, help='Saves the current homing values'
    # )
    #
    # parser.add_argument(
    #     '--laser_calibrate', '-c',
    #     action='store_true', default=False, help='Calibrate lasers'
    # )
    #
    # parser.add_argument(
    #     '--laser_sample', '-l',
    #     action='store_true', default=False, help='Get single sample from laser'
    # )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true', help='Increases the output verbosity')

    args, leftovers = parser.parse_known_args()

    return parser.parse_args()


if __name__ == '__main__':

    # Load camera parameters
    while int(load_json_config()['source']) == -1:
        setup_source()

    ns = parse_args()

    if ns.calibrate:  # Perform camera calibration
        while int(load_json_config()['source']) == -1:
            setup_source()
        cal = Calibration(load_json_config())
        cal.calibrate_from_images()
        print('a')

    elif ns.source:
        setup_source()
        while int(load_json_config()['source']) == -1:
            setup_source()

    else:
        logging.error('Please specify program mode.')
        raise SystemExit
