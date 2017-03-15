from camera_utils import CameraCalibrator, ImageUndistorter

def calibrate_camera_and_undistort():
    calbrator = CameraCalibrator(
        calibration_image_glob='../camera_cal/calibration*.jpg',
        nx=9,
        ny=6
    )
    mtx, dist = calbrator.calibrate()

    undistorter = ImageUndistorter(mtx, dist)
    undistorter.undistort_and_save(
        img_path='../test_images/straight_lines1.jpg',
        out_path='../output_images/straight_lines1_undistort.jpg'
    )
    undistorter.undistort_and_save(
        img_path='../camera_cal/calibration1.jpg',
        out_path='../output_images/calibration1_undistort.jpg'
    )


if __name__ == '__main__':
    calibrate_camera_and_undistort()
