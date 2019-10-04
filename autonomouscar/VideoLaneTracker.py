import imutils
import cv2
import datetime
from .VideoFrameLaneProcessor import VideoFrameLaneProcessor


class VideoLaneTracker:

    max_rolling_road_feature_locations = 3

    source_video_file = None
    source_video = None
    contour_outline_color_hsv = (0, 255, 0)
    state_waiting = "waiting"
    state_tracking = "tracking"

    # direction_foreward = "forward"
    # direction_backward = "backward"
    cm_per_pixel = 30  # Fixed value for now
    min_road_line_area = 20

    def __init__(self, source_video_file):
        self.source_video_file = source_video_file

    def open_video(self):
        self.source_video = cv2.VideoCapture(self.source_video_file)
        if self.source_video.isOpened is False:
            raise Exception("Error opening video file")

    def track_lanes(self):
        frame_processor = VideoFrameLaneProcessor()
        frame_resize_ratio = None

        tracking_state = self.state_waiting
        initial_time_s = None
        last_y = 0
        initial_y = 0
        speed_kph = 0.0

        while self.source_video.isOpened() is True:
            print("============ FRAME ===========")
            current_time_s = datetime.datetime.now()
            return_value, source_frame = self.source_video.read()
            if frame_resize_ratio is None:
                frame_resize_ratio = frame_processor.get_frame_resize_ratio(
                    source_frame
                )
            if return_value is True:
                road_frame = frame_processor.get_birdseye_view_of_road(
                    source_frame
                )
                merged_frame = source_frame

                resized_road_frame = imutils.resize(road_frame, width=300)
                threshold_frame = frame_processor.get_threshold_frame(
                    resized_road_frame
                )

                road_features = frame_processor.get_road_features(
                    threshold_frame
                )

                # loop over the contours
                biggest_area = 0
                was_motion_found = False
                for road_feature in road_features:

                    # multiply the contour (x, y)-coordinates
                    # by the resize ratio,
                    # then draw the contours and the
                    # name of the shape on the image
                    '''
                    resized_contour = frame_processor.get_resized_contour(
                        road_feature,
                        frame_resize_ratio
                    )
                    '''
                    cv2.drawContours(
                        road_frame,
                        [road_feature],
                        -1,
                        self.contour_outline_color_hsv,
                        2
                    )
                    '''
                    cv2.putText(
                        image,
                        shape,
                        (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
                    '''
                    was_motion_found = self.was_motion_found(
                        road_feature,
                        biggest_area)
                    x, y, w, h = cv2.boundingRect(
                        road_feature
                    )
                    if was_motion_found is True:
                        if tracking_state == self.state_waiting:
                            tracking_state = self.state_tracking
                            biggest_area = w * h
                            initial_time_s = current_time_s
                            initial_y = y
                            last_y = y
                        else:
                            seconds_since_last_frame = \
                                self.get_seconds_since_last_frame(
                                    initial_time_s,
                                    current_time_s
                                )
                            if tracking_state == self.state_tracking:
                                if y >= last_y:
                                    # direction = self.direction_foreward
                                    absolute_change = y + h - initial_y
                                else:
                                    # direction = self.direction_backward
                                    absolute_change = initial_y - y
                                speed_kph = self.get_speed_kph(
                                    absolute_change,
                                    self.cm_per_pixel,
                                    seconds_since_last_frame
                                )
                                cv2.rectangle(
                                    road_frame,
                                    (x, y), (w, h),
                                    self.contour_outline_color_hsv,
                                    2
                                )
                                print(speed_kph)

                    merged_frame = self.overlay_hud(
                        source_frame,
                        threshold_frame
                    )
                    self.overlay_stats(merged_frame, speed_kph)
                    # show the output image
                    cv2.imshow("Image", merged_frame)
                    # cv2.imshow("Threshold", threshold_frame)

                    # roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
                    # cv2.imshow('Video', source_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break

        self.source_video.release()
        cv2.destroyAllWindows()

    def overlay_hud(self, source_frame, road_frame):
        for color in range(0, 3):
            source_frame[
                0:road_frame.shape[0],
                0:road_frame.shape[1],
                color
            ] = road_frame
        return source_frame

    def overlay_stats(self, source_frame, speed_kph):
        cv2.rectangle(
            source_frame,
            (0, 0),
            (source_frame.shape[1], 40),
            (0, 0, 0),
            thickness=-1
        )
        speed_string = 'Estimated Speed: {:06.2f} km/h'.format(speed_kph)
        cv2.putText(
            source_frame,
            speed_string,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def was_motion_found(self, road_line_contour, biggest_area):
        was_motion_found = False
        (x1, y1, w1, h1) = cv2.boundingRect(road_line_contour)
        road_line_area = w1 * h1

        if road_line_area < 800 or road_line_area > 1400:
            return False

        print("w: " + str(w1) + ", h: " + str(h1))

        # set the dimensions of the road lines
        if h1 < w1 * 5:
            return False
        if h1 > w1 * 7:
            return False
        if w1 > 20:
            return False
        if h1 < 50:
            return False

        if road_line_area > self.min_road_line_area and \
                road_line_area > biggest_area:
            was_motion_found = True

        # peri = cv2.arcLength(road_line_contour, True)
        # vertices = cv2.approxPolyDP(road_line_contour, 0.04 * peri, True)
        # if len(vertices) != 4:
        #     was_motion_found = False

        return was_motion_found

    def get_speed_kph(self, pixel_delta, cm_per_pixel, time_delta_s):
        if time_delta_s > 0.0:
            return ((pixel_delta * cm_per_pixel) / time_delta_s) / 1000
        else:
            return 0.0

    # calculate elapsed seconds
    def get_seconds_since_last_frame(self, start_time_s, end_time_s):
        time_diff_s = (end_time_s - start_time_s).total_seconds()
        return time_diff_s
