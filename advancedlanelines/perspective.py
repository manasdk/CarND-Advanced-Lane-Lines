import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self, img_size):
        print(img_size)
        self.src_points, self.dest_points = self._get_src_dst(img_size)
        print(self.src_points)
        print(self.dest_points)
        self.img_size = img_size
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)

    @classmethod
    def warp_perspective_and_save(cls, undist_img_path, out_path):
        """
        Given path to an undistrorted image will create a PerspectiveTransformer and save generated
        image to  out_path.
        """
        undist_img = cv2.imread(undist_img_path)
        perspective_transformer = PerspectiveTransformer(
            img_size=(undist_img.shape[1], undist_img.shape[0])
        )
        warp_perspective_img = perspective_transformer.warp_perspective(undist_img)
        cv2.imwrite(out_path, warp_perspective_img)

    def warp_perspective(self, undist_img):
        """
        warp the perspective based on previously computed perspective transformer.
        """
        return cv2.warpPerspective(undist_img, self.M, self.img_size)

    def _get_src_dst(self, img_size):
        """
        Making some assumptions about our test set performing a transform
        """
        # picked some arbitrary points
        # src = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])

        # picking roughly a trapezoid that would contain a lane
        img_x = img_size[0]
        img_y = img_size[1]
        # some trial and error
        # src = np.float32([
        #     [img_x * 0.2, img_y],
        #     [int(img_x * 0.466), img_y * 0.625],
        #     [int(img_x * 0.535), img_y * 0.625],
        #     [img_x * 0.85, img_y],
        # ])
        # for 1280 x 720 image src would be
        # [[  256.   720.]
        #  [  596.   450.]
        #  [  684.   450.]
        #  [ 1088.   720.]]
        # for 1280 x 720 image dst is desired to be
        # [[  100.   620.]
        # [  100.   100.]
        # [ 1180.   100.]
        # [ 1180.   620.]]
        # x = 100
        # dst = np.float32([
        #     [x, img_size[1]-x],
        #     [x, x],
        #     [img_size[0]-x, x],
        #     [img_size[0]-x, img_size[1]-x]
        # ])

        # picking roughly a trapezoid that would contain a lane. the coefficients are achieved
        # with some trial and error
        src = np.float32([
            [img_x * 0.2, img_y],
            [int(img_x * 0.379), img_y * 0.75],
            [int(img_x * 0.61), img_y * 0.75],
            [img_x * 0.85, img_y],
        ])
        dst = np.float32([
            [src[1][0], src[0][1]],
            src[1],
            src[2],
            [src[2][0], src[3][1]]
        ])
        return src, dst
