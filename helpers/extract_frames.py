import cv2
import os
import sys
from tqdm import tqdm


def extract_frames(video_path, output_folder, frame_interval=1):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame counter
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        # Extract a frame at the specified interval
        if frame_count % frame_interval == 0:
            # Generate the output filename
            output_filename = f"frame_{frame_count:06d}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # Save the frame as an image
            cv2.imwrite(output_path, frame)

            print(f"Extracted frame: {output_filename}")

        frame_count += 1

    # Release the video capture object
    video.release()

    print(f"Extraction complete. Total frames: {total_frames}")


def extract_frame(video_path, output_file):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the middle frame index
    middle_frame_index = total_frames // 2

    # Set the video position to the middle frame
    video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # Read the middle frame
    ret, frame = video.read()

    if ret:
        # Save the middle frame as an image
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_file, frame)
        # print(f"Extracted middle frame: {output_file}")
    else:
        print("Failed to extract the middle frame")

    # Release the video capture object
    video.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py function_name ...function_params...")
        sys.exit(1)

    function_name = sys.argv[1]

    if function_name == "extract_frames":
        if len(sys.argv) != 5:
            print(
                "Usage: python extract_frames.py extract_frames video_path output_folder frame_interval")
            sys.exit(1)
        video_path = sys.argv[2]
        output_folder = sys.argv[3]
        frame_interval = int(sys.argv[4])
        extract_frames(video_path, output_folder, frame_interval)

    elif function_name == "extract_frame":
        if len(sys.argv) != 4:
            print("Usage: python extract_frames.py extract_frame video_path output_file")
            sys.exit(1)
        video_path = sys.argv[2]
        output_file = sys.argv[3]
        extract_frame(video_path, output_file)

    elif function_name == "extract_frame_all":
        if len(sys.argv) != 4:
            print(
                "Usage: python extract_frames.py extract_frames_all video_folder output_folder")
            sys.exit(1)
        video_folder = sys.argv[2]
        output_folder = sys.argv[3]
        for video_file in tqdm(os.listdir(video_folder)):
            video_path = os.path.join(video_folder, video_file)
            extract_frame(video_path, output_folder
                          + "/" + video_file + ".jpg")

    else:
        print(f"Unknown function: {function_name}")
        print("Available functions: extract_frames, extract_frame")
        sys.exit(1)
