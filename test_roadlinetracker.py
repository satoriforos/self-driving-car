#!/usr/bin/env python3

from autonomouscar.VideoLaneTracker import VideoLaneTracker


def main():
    video_file = 'train.mp4'
    lane_tracker = VideoLaneTracker(video_file)
    lane_tracker.open_video()
    lane_tracker.track_lanes()


if __name__ == "__main__":
    main()
