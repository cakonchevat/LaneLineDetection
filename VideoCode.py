from moviepy.editor import VideoFileClip
from FinalCode import lane_finding_pipeline


def process_video(input_file, output_file):
    clip = VideoFileClip(input_file)

    clip = clip.set_fps(15)

    processed_clip = clip.fl_image(lane_finding_pipeline)
    processed_clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    input_video = "solidWhiteRight.mp4"  # Input video file
    output_video = "solution.mp4"    # Output video file

    process_video(input_video, output_video)
