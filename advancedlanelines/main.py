from camera_utils import CameraCalibrator, ImageUndistorter
from image_binary import ImageBinarizer
from perspective import PerspectiveTransformer
from lane import LaneExtractor


def calibrate_camera_and_undistort():
    """
    Calibrates camera and saves some undistorted images
    """
    calbrator = CameraCalibrator(
        calibration_image_glob='../camera_cal/calibration*.jpg',
        nx=9,
        ny=6
    )
    mtx, dist = calbrator.calibrate()

    undistorter = ImageUndistorter(mtx, dist)
    undistorter.undistort_given_path_and_save(
        img_path='../test_images/straight_lines1.jpg',
        out_path='../output_images/straight_lines1_undistort.jpg'
    )
    undistorter.undistort_given_path_and_save(
        img_path='../camera_cal/calibration1.jpg',
        out_path='../output_images/calibration1_undistort.jpg'
    )


def binarize():
    """
    Binarizes images, runs on some undistorted images and saves those undistorted images.
    """
    binarizer = ImageBinarizer(
        thresh_min=20,
        thresh_max=100,
        s_thresh_min=170,
        s_thresh_max=255
    )
    binarizer.binarize_given_path_and_save(
        img_path='../output_images/straight_lines1_undistort.jpg',
        out_path='../output_images/straight_lines1_undistort_binary.jpg'
    )


def perspective_transform():
    perspective_transformer = PerspectiveTransformer.warp_perspective_and_save(
        undist_img_path='../output_images/straight_lines1_undistort.jpg',
        out_path='../output_images/straight_lines1_undistort_warped.jpg'
    )
    perspective_transformer = PerspectiveTransformer.warp_perspective_and_save(
        undist_img_path='../output_images/straight_lines1_undistort_binary.jpg',
        out_path='../output_images/straight_lines1_undistort_binary_warped.jpg'
    )


def extract_lane():
    extractor = LaneExtractor()
    extractor.extract_lane_from_scratch_and_save(
        binary_warped_path='../output_images/straight_lines1_undistort_binary_warped.jpg',
        out_path='../output_images/straight_lines1_undistort_binary_warped_lanes.jpg'
    )


if __name__ == '__main__':
    # calibrate_camera_and_undistort()
    # binarize()
    perspective_transform()
    extract_lane()
