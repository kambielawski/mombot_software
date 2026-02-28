import os
import cv2
import time
from datetime import datetime

MAX_WIDTH = 4096
MAX_HEIGHT = 2160

# Extract timestamps and sort files


def get_timestamp_from_image_path(image_path):
    # Extract timestamp portion from UUID-YYYY-MM-DDThh.mm.ss.nnnnnn format
    image_file_name = image_path.split('/')[-1]
    timestamp_str = image_file_name.split(
        '-', 5)[-1].replace('.jpg', '').replace('.png', '')
    # Try parsing with microseconds first, then without
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H.%M.%S.%f')
    except ValueError:
        # Fall back to parsing without microseconds
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H.%M.%S')


def get_timestamp_from_command_file(timestamp_str):
    # Extract timestamp portion from UUID-YYYY-MM-DDThh:mm:ss.nnnnnnZ format
    timestamp_str = timestamp_str.replace('Z', '')
    # Try parsing with microseconds first, then without
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        # Fall back to parsing without microseconds
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')


def get_timestamp(input_str):
    if input_str.endswith('.jpg') or input_str.endswith('.png'):
        return get_timestamp_from_image_path(input_str)
    else:
        return get_timestamp_from_command_file(input_str)


def sort_images(images_path):
    # Get all image files
    image_files = [f for f in os.listdir(
        images_path) if f.endswith(('.jpg', '.png'))]

    # Extract timestamps and create (filename, timestamp) pairs
    timestamp_pairs = []
    for filename in image_files:
        # Extract the timestamp part (assumes format like your example)
        timestamp_str = '-'.join(filename.split('-')[-3:])[:-4]
        # Convert to datetime object
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H.%M.%S")
        timestamp_pairs.append((filename, timestamp))

    # Sort by timestamp
    timestamp_pairs.sort(key=lambda x: x[1])

    if not timestamp_pairs:
        return []

    # Group images into chunks
    chunks = []
    current_chunk = [timestamp_pairs[0][0]]  # Start with first image
    last_timestamp = timestamp_pairs[0][1]

    for filename, timestamp in timestamp_pairs[1:]:
        time_diff = (timestamp - last_timestamp).total_seconds()

        if time_diff > 60.0:  # If gap is more than 4 seconds
            chunks.append(current_chunk)  # Save current chunk
            current_chunk = [filename]  # Start new chunk
        else:
            current_chunk.append(filename)

        last_timestamp = timestamp

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def assemble_video(images_path: str, videos_path: str, file_names):
    start = time.time()
    if not file_names:
        return

    # Get full paths for all files
    image_paths = sorted([os.path.join(images_path, f) for f in file_names])

    # Get start and end timestamps from first and last filenames
    start_timestamp = '-'.join(image_paths[0].split('-')[-3:])[:-4]
    end_timestamp = '-'.join(image_paths[-1].split('-')[-3:])[:-4]

    # # Calculate duration in seconds
    # start_time = datetime.strptime(start_timestamp, '%Y-%m-%dT%H.%M.%S.%f')
    # end_time = datetime.strptime(end_timestamp, '%Y-%m-%dT%H.%M.%S.%f')
    start_time = get_timestamp(image_paths[0])
    end_time = get_timestamp(image_paths[-1])
    duration = (end_time - start_time).total_seconds()

    # Create video filename
    video_name = f"{start_timestamp}_{end_timestamp}_{duration:.1f}_seconds.mp4"
    video_path = os.path.join(videos_path, video_name)

    if os.path.exists(video_path):
        return video_path

    # Create video writer
    first_img = cv2.imread(image_paths[0])
    height, width = first_img.shape[:2]

    # print("Original image dimensions:", height, width)

    # # Scale down if needed while maintaining aspect ratio
    # if width > MAX_WIDTH or height > MAX_HEIGHT:
    #     scale = min(MAX_WIDTH/width, MAX_HEIGHT/height)
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    #     # print(f"Scaling down to: {new_width}x{new_height}")
    #     width, height = new_width, new_height

    # Create videos directory if it doesn't exist
    videos_path = os.path.join(os.path.dirname(images_path), 'videos')
    os.makedirs(videos_path, exist_ok=True)

    # Initialize video writer with scaled dimensions
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))

    # Add each image to video, resizing if necessary
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()

    end = time.time()

    func_runtime = end - start

    print(f"Created video from {len(file_names)} images: {video_name} in {func_runtime:.1f} seconds")
    return video_path


def assemble_video_from_time_range(images_path: str,
                                   videos_path: str,
                                   start_timestamp: str,
                                   end_timestamp: str):
    if not images_path:
        return

    # Calculate duration in seconds
    start_time = get_timestamp(start_timestamp)
    end_time = get_timestamp(end_timestamp)

    # Wait for all images to sync from Dropbox
    while True:
        # Get full paths for all files
        image_paths = [os.path.join(images_path, f) for f in os.listdir(
            images_path) if f.endswith(('.jpg', '.png'))]
        # Get the most recent image timestamp
        image_timestamps = [get_timestamp(img_path)
                            for img_path in image_paths]
        if not image_timestamps:
            print("[Medea] No images found, waiting...")
            time.sleep(3.0)
            continue

        most_recent_time = max(image_timestamps)

        # If we've exceeded the end time, we can proceed
        if most_recent_time >= end_time:
            break

        # Otherwise, check if we're within 2.5 seconds
        time_difference = abs((most_recent_time - end_time).total_seconds())
        if time_difference <= 2.5:
            break

        print(f"[Medea] Waiting for images to sync from Dropbox...")
        print(f"[Medea] End time: {end_time}")
        print(f"[Medea] Most recent image time: {most_recent_time}")
        print(f"[Medea] Time difference: {time_difference:.1f} seconds")
        time.sleep(3.0)

    # Filter only images within the time range
    image_paths = [img_path for img_path in image_paths
                   if (start_time <= get_timestamp(img_path) <= end_time)]

    return assemble_video(images_path=images_path,
                          videos_path=videos_path,
                          file_names=image_paths)

    # # Create video writer
    # first_img = cv2.imread(image_paths[0])
    # height, width = first_img.shape[:2]

    # # print("Original image dimensions:", height, width)

    # # Scale down if needed while maintaining aspect ratio
    # if width > MAX_WIDTH or height > MAX_HEIGHT:
    #     scale = min(MAX_WIDTH/width, MAX_HEIGHT/height)
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    #     # print(f"Scaling down to: {new_width}x{new_height}")
    #     width, height = new_width, new_height

    # # Create videos directory if it doesn't exist
    # videos_path = os.path.join(os.path.dirname(images_path), 'videos')
    # os.makedirs(videos_path, exist_ok=True)

    # # Create video filename
    # video_name = f"{start_timestamp}_{end_timestamp}_{duration:.1f}_seconds.mp4"
    # video_path = os.path.join(videos_path, video_name)

    # # Initialize video writer with scaled dimensions
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))

    # # Add each image to video, resizing if necessary
    # for img_path in image_paths:
    #     frame = cv2.imread(img_path)
    #     if frame.shape[:2] != (height, width):
    #         frame = cv2.resize(frame, (width, height))
    #     out.write(frame)

    # out.release()

    # print(f"Created video from {len(image_paths)} images: {video_name}")
    # return video_path
