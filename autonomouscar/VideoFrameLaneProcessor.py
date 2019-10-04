import numpy as np
import cv2
import imutils


class VideoFrameLaneProcessor:

    perspective_frame_size = (200, 200)

    def get_birdseye_view_of_road(self, source_frame):
        '''
        This should be calculated by extrapolating the lane lines
        and finding the slope of those intersecting lines.
        But for now I'm hard coding the values.
        '''
        source_perspective_shape = np.float32([
            [258, 254], [360, 252],
            [88, 359], [532, 359]
        ])
        destination_perspective_shape = np.float32([
            [0, 0], [300, 0],
            [0, 300], [300, 300]
        ])
        road_perpective_shift = cv2.getPerspectiveTransform(
            source_perspective_shape,
            destination_perspective_shape
        )
        birdseye_frame = cv2.warpPerspective(
            source_frame,
            road_perpective_shift,
            self.perspective_frame_size
        )
        return birdseye_frame

    def get_threshold_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        contrasted_frame = cv2.equalizeHist(blurred_frame)
        threshold_frame = cv2.threshold(
            contrasted_frame,
            252,
            255,
            cv2.THRESH_BINARY
        )[1]
        '''
        threshold_frame = cv2.threshold(
            blurred,
            80,
            255,
            cv2.THRESH_BINARY
        )[1]
        '''
        '''
        threshold_frame = cv2.adaptiveThreshold(
            blurred_frame,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        '''
        '''
        threshold_frame = cv2.threshold(
            blurred_frame,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        '''
        return threshold_frame

    def get_road_features(self, frame):
        # find features in the thresholded image and initialize the
        # shape detector
        features = cv2.findContours(
            frame.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        features = features[0] if imutils.is_cv2() else features[1]
        return features

    def get_frame_resize_ratio(self, frame, frame_width=300):
        resized_frame = imutils.resize(frame, width=frame_width)
        ratio = frame.shape[0] / float(resized_frame.shape[0])
        return ratio

    def get_resized_contour(self, contour, frame_resize_ratio):
        resized_contour = contour.astype("float")
        resized_contour *= frame_resize_ratio
        resized_contour = resized_contour.astype("int")
        return resized_contour
