import cv2
import numpy as np
import os.path
import pickle
from moviepy.editor import VideoFileClip

from camera_utils import CameraCalibrator, ImageUndistorter
from image_binary import ImageBinarizer
from perspective import PerspectiveTransformer
from lane import LaneExtractor


class Pipeline():
    camera_calibration_data_file = 'camera.dat'

    def __init__(self):
        self.calibrator = CameraCalibrator(
            calibration_image_glob='../camera_cal/calibration*.jpg',
            nx=9,
            ny=6
        )
        mtx, dist = self._calibrate_camera()
        self.undistorter = ImageUndistorter(mtx, dist)
        self.binarizer = ImageBinarizer(
            thresh_min=20,
            thresh_max=100,
            s_thresh_min=170,
            s_thresh_max=255
        )
        self.perspective_transformer = None
        self.lane_extractor = LaneExtractor()

    def process_video(self, input_video_path, output_video_path):
        input_clip = VideoFileClip(input_video_path)
        output_clip = input_clip.fl_image(self.process_single_image)
        output_clip.write_videofile(output_video_path, audio=False)

    def process_single_image(self, img):
        # 1. undistort the image
        undist_img = self.undistorter.undistort(img)
        # 2. binarize the image
        binary_img = self.binarizer.binarize(undist_img)
        if not self.perspective_transformer:
            self._init_perspective_transformer(undist_img)
        # 3. warp the binarized image
        binary_warped_img = self.perspective_transformer.warp_perspective(binary_img)
        # 4. extract lanes, image needs to be gray scale for lane extraction
        binary_warped_img = cv2.cvtColor(binary_warped_img, cv2.COLOR_RGB2GRAY)
        left_fitx, _, right_fitx, _ = self.lane_extractor.extract_lane(binary_warped_img)
        # 5. fill lane poly
        filled_binary_warped_img = self._fill_lane_poly(binary_warped_img, left_fitx, right_fitx)
        return self._superimpose_filled_on_original(
            org_img=img, filled_binary_warped_img=filled_binary_warped_img
        )

    def _init_perspective_transformer(self, undist_img):
        self.perspective_transformer = PerspectiveTransformer(
            img_size=(undist_img.shape[1], undist_img.shape[0])
        )

    def _fill_lane_poly(self, binary_warped_img, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, binary_warped_img.shape[0] - 1, binary_warped_img.shape[0])

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        return color_warp


    def _superimpose_filled_on_original(self, org_img, filled_binary_warped_img):
        # unawarp filled_binary_warped_img
        filled_binary_unwarped_img = self.perspective_transformer.warp_perspective_inverse(
            filled_binary_warped_img
        )
        # super-impose unwarped on original image
        return cv2.addWeighted(org_img, 1, filled_binary_unwarped_img, 0.3, 0)

    def _calibrate_camera(self):
        if os.path.isfile(self.camera_calibration_data_file):
            calibration_data = pickle.load(open(self.camera_calibration_data_file, "rb"))
            return calibration_data['mtx'], calibration_data['dist']
        mtx, dist = self.calibrator.calibrate()
        pickle.dump({'mtx': mtx, 'dist': dist}, open(self.camera_calibration_data_file, "wb" ))
        return mtx, dist


if __name__ == '__main__':
    """
    Test method for process_single_image of Pipeline
    """
    pipeline = Pipeline()
    img = cv2.imread('../test_images/straight_lines1.jpg')
    out_img = pipeline.process_single_image(img)
    cv2.imwrite('../output_images/straight_lines1_lane_found.jpg', out_img)
    out_img = pipeline.process_single_image(img)
