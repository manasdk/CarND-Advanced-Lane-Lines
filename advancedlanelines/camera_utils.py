import glob

import cv2
import numpy as np


class CameraCalibrator:
    """
    Calibrates the camera and based on supplied calibration images will compute the camera matrix
    and distance coefficients
    """
    def __init__(self, calibration_image_glob, nx, ny):
        self.calibration_images = glob.glob(calibration_image_glob)
        self.nx = nx
        self.ny = ny

    def calibrate(self):
        """
        calibrates camera based on the calibration images

        :return: the camera matrix and distance coeffecients
        """
        objpoints, imgpoints, img_shape = self._find_objpoints_imgpoints()

        # calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

        return mtx, dist

    def _find_objpoints_imgpoints(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = []

        for _, calibration_image in enumerate(self.calibration_images):
            img = cv2.imread(calibration_image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints, gray.shape[::-1]


class ImageUndistorter:
    """
    undistorts images based on the camera matrix and distance coefficients
    """
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def undistort_given_path(self, img_path):
        img = cv2.imread(img_path)
        return self.undistort(img=cv2.imread(img_path))

    def undistort_given_path_and_save(self, img_path, out_path):
        undistorted_img = self.undistort(img_path=img_path)
        cv2.imwrite(out_path, undistorted_img)
        return undistorted_img
