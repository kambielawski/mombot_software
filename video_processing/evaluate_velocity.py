"""
Takes in path to video: *.mp4 and evaluates the objective function of velocity a video with a single bot
"""

import logging
import time
import cv2
import numpy as np

def crop_image(frame, crop_top, crop_bottom, crop_left, crop_right):
    height, width, _ = frame.shape
    cropped_frame = frame[crop_top:height - crop_bottom, crop_left:width-crop_right]
    return cropped_frame

def downsample_frame(frame, dimension=128):
    height, width = frame.shape[:2]

    # Input image needs to already be square (cropped to square)
    assert height == width

    # Resize
    frame = cv2.resize(
        frame, (dimension, dimension))

    return frame


def remove_isolated_pixels_simple(binary_image):
    # Label connected components
    num_labels, labels = cv2.connectedComponents(binary_image, connectivity=8)

    # Count pixels in each component
    output = binary_image.copy()

    for label in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label).astype(np.uint8) * 255
        pixel_count = np.sum(component_mask) // 255

        if pixel_count == 1:  # Isolated single pixel
            output[labels == label] = 0

    return output

def mask_frame_adaptive(frame, 
                        block_size=11, 
                        C=2):
    """
    Adaptive thresholding!
    For each pixel:
        - consider a block x block square around the pixel.
        - calculate the mean of the block
        - subtract a constant C from the mean
        - if the pixel is greater than the mean, set it to 255, otherwise set it to 0
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold adjusts to local image characteristics
    mask = cv2.adaptiveThreshold(gray,  # image
                                 255,  # max value
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # adaptive method
                                 cv2.THRESH_BINARY_INV,  # threshold type
                                 block_size,  # block size
                                 C)  # C value
    return mask

def get_centroids_and_areas(binary_image):
    """
    Get the centroids sorted by area (largest first).
    Returns:
        List[Tuple[int,int]] - centroids sorted largest-to-smallest
    """
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,  # only external contours
        cv2.CHAIN_APPROX_SIMPLE)

    centroids_with_area = []
    for contour in contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids_with_area.append(((cx, cy), area))

    # Sort by area descending and return only centroids
    centroids_with_area.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in centroids_with_area]

def identify_bot_in_frame(frame, 
                          crop_top=0, 
                          crop_bottom=0,
                          crop_left=1358, 
                          crop_right=850,
                          block_size=17, 
                          C=29, 
                          dimension=512, 
                          mask_radius=240,
                          visualize: bool = False) -> tuple:
    """
    Identify the bot in the frame and return its coordinates
    """

    # Crop the frame
    cropped_frame = crop_image(frame, crop_top, crop_bottom, crop_left, crop_right)

    # Downsample the frame
    downsampled_frame = downsample_frame(cropped_frame, dimension=dimension)

    # Thresholding to create a binary mask
    masked_frame = mask_frame_adaptive(downsampled_frame, block_size=block_size, C=C)

    # Post-process masked frame (remove noise, apply circular mask)
    frame = remove_isolated_pixels_simple(masked_frame)
    frame = apply_circular_mask(frame, center=(dimension//2, dimension//2), radius=mask_radius)

    # Compute the centroids and areas of all detected blobs
    centroids = get_centroids_and_areas(frame)

    # Return the largest centroid (assuming it's the bot)
    return (centroids[0] if centroids else None), frame


def apply_circular_mask(binary_image, center, radius):
    """
    Keep only the content inside a circular region, zero out everything else.

    Parameters:
    -----------
    binary_image : np.ndarray
        Your binary mask (0s and 255s)
    center : tuple
        (x, y) coordinates of circle center in pixels
    radius : int
        Radius of the circle in pixels

    Returns:
    --------
    masked_image : np.ndarray
        Binary image with only circular region preserved
    """
    # Create a blank mask (all zeros)
    h, w = binary_image.shape[:2]
    circular_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a filled white circle on the mask
    cv2.circle(circular_mask, center, radius, 255, -1)  # -1 means filled

    # Apply the circular mask using bitwise AND
    masked_image = cv2.bitwise_and(binary_image, circular_mask)

    return masked_image


def compute_centroids(frames: list, 
                      crop_top=0,
                    crop_bottom=0,
                    crop_left=1358,
                    crop_right=850,
                    block_size=17,
                    C=29,
                    dimension=512,
                    mask_radius=240, 
                    visualize: bool = False) -> list:
    """
    Process frames to extract centroids
    """
    centroids = []
    vis_frames = []  # For visualization
    for frame in frames:
        point, vis_frame = identify_bot_in_frame(frame, crop_top,
                    crop_bottom,
                    crop_left,
                    crop_right,
                    block_size,
                    C,
                    dimension,
                    mask_radius, 
                    visualize)
        if point: 
            centroids.append(point)
            vis_frames.append(vis_frame)

    return centroids, vis_frames

def wait_until_opencv_can_read(path, retry_delay=5.0):
    while True:
        cap = cv2.VideoCapture(path)
        
        # Opened doesn't mean it's readable — you must test cap.read()
        ret, frame = cap.read()
        
        if ret:
            cap.release()
            return True
        
        cap.release()
        time.sleep(retry_delay)

def extract_frames(video_path: str) -> list:
    """
    Extract frames from video
    """
    wait_until_opencv_can_read(video_path)
    logging.info("Reading video from path: %s", video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def compute_velocity(centroids: list, dimension: int, total_seconds: float) -> float:
    """
    Compute the velocity (in pixels/s) between each image
    """

    print(centroids)

    displacements = []
    for i in range(1, len(centroids)):
        point = centroids[i]
        point_prev = centroids[i-1]
        displacement = np.sqrt((point[0] - point_prev[0])**2 + (point[1] - point_prev[1])**2)
        if displacement <= dimension / 20: # Ignore large jumps
            displacements.append(displacement)
    velocity = np.sum(displacements) / total_seconds
    return velocity

def evaluate_velocity(video_path: str,
                      output_path: str,
                      visualize: bool = False,
                      dt_seconds: float = 5.0,
                    crop_top=0,
                    crop_bottom=0,
                    crop_left=1358,
                    crop_right=850,
                    block_size=17,
                    C=29,
                    dimension=512,
                    mask_radius=240) -> float:
    """
    Evaluates the objective function: velocity of trajectory from the predicted video
    """

    # 1. Extract frames from video
    frames = extract_frames(video_path)
    n_frames = len(frames)

    # 2. Process frames to extract centroids
    centroids, vis_frames = compute_centroids(frames,
                                  crop_top,
                                crop_bottom,
                                crop_left,
                                crop_right,
                                block_size,
                                C,
                                dimension,
                                mask_radius)

    # if visualize:
    #     (vis_frames)

    # 3. Compute velocities between consecutive centroids
    velocity = compute_velocity(centroids, dimension, total_seconds=n_frames * dt_seconds)

    return velocity


def parse_image_timestamp(filename):
    """
    Parse timestamp from image filename.
    Image filenames: UUID-YYYY-MM-DDThh.mm.ss.nnnnnn.png (dots for time, no Z)
    """
    from datetime import datetime
    timestamp_str = filename.split('-', 5)[-1].replace('.jpg', '').replace('.png', '')
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H.%M.%S.%f')
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H.%M.%S')
        except ValueError:
            return None


def parse_command_timestamp(ts_str):
    """
    Parse timestamp from command file format.
    Command timestamps: YYYY-MM-DDThh:mm:ss.nnnnnnZ (colons for time, with Z)
    """
    from datetime import datetime
    ts_str = ts_str.replace('Z', '')
    try:
        return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')


def get_images_with_datetimes(images_path: str,
                               start_timestamp: str,
                               end_timestamp: str) -> list:
    """
    Get images within a time range, sorted by datetime.

    Args:
        images_path: Path to directory containing images
        start_timestamp: Start time in command format (YYYY-MM-DDThh:mm:ss.nnnnnnZ)
        end_timestamp: End time in command format

    Returns:
        List of (image_path, datetime) tuples sorted by datetime
    """
    import os

    start_dt = parse_command_timestamp(start_timestamp)
    end_dt = parse_command_timestamp(end_timestamp)

    image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg'))]

    images_with_datetimes = []
    for filename in image_files:
        ts = parse_image_timestamp(filename)
        if ts and start_dt <= ts <= end_dt:
            full_path = os.path.join(images_path, filename)
            images_with_datetimes.append((full_path, ts))

    # Sort by datetime
    images_with_datetimes.sort(key=lambda x: x[1])

    return images_with_datetimes


def get_centroids_and_datetimes(images_with_datetimes: list,
                                 crop_top=0,
                                 crop_bottom=0,
                                 crop_left=1358,
                                 crop_right=850,
                                 block_size=17,
                                 C=29,
                                 dimension=512,
                                 mask_radius=240) -> tuple:
    """
    Get centroids and datetimes for images where a bot was detected.

    Args:
        images_with_datetimes: List of (image_path, datetime) tuples
        crop_*, block_size, C, dimension, mask_radius: Image processing params

    Returns:
        Tuple of (centroids, datetimes) where each list only includes
        images where a centroid was successfully detected
    """
    centroids = []
    datetimes = []

    for image_path, dt in images_with_datetimes:
        frame = cv2.imread(image_path)
        if frame is None:
            logging.warning("Could not read image: %s", image_path)
            continue

        point, _ = identify_bot_in_frame(
            frame,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            crop_right=crop_right,
            block_size=block_size,
            C=C,
            dimension=dimension,
            mask_radius=mask_radius
        )

        if point is not None:
            centroids.append(point)
            datetimes.append(dt)

    return centroids, datetimes


def compute_velocities_over_time(centroids: list,
                                  datetimes: list,
                                  max_velocity: float = 5.0) -> tuple:
    """
    Compute velocity (in pixels/s) between consecutive centroids using actual timestamps.

    Args:
        centroids: List of (x, y) centroid coordinates
        datetimes: List of datetime objects corresponding to each centroid
        max_velocity: Maximum plausible velocity (pixels/s). Higher values are filtered
                      as tracking artifacts.

    Returns:
        Tuple of (velocities, delta_times) where velocities are in pixels/s
        and delta_times are in seconds
    """
    assert len(centroids) == len(datetimes), "centroids and datetimes must have same length"

    if len(centroids) <= 1:
        return [], []

    velocities = []
    delta_times = []

    for i in range(1, len(centroids)):
        point = centroids[i]
        point_prev = centroids[i - 1]
        t = datetimes[i]
        t_prev = datetimes[i - 1]

        # Compute displacement in pixels
        displacement = np.sqrt(
            (point[0] - point_prev[0])**2 + (point[1] - point_prev[1])**2
        )

        # Compute time delta in seconds
        dt = (t - t_prev).total_seconds()

        if dt <= 0:
            continue

        velocity = displacement / dt

        # Filter out implausibly high velocities (tracking artifacts/jumps)
        if velocity <= max_velocity:
            velocities.append(velocity)
            delta_times.append(dt)

    return velocities, delta_times


def evaluate_velocity_from_images(images_path: str,
                                   start_timestamp: str,
                                   end_timestamp: str,
                                   crop_top=0,
                                   crop_bottom=0,
                                   crop_left=1358,
                                   crop_right=850,
                                   block_size=17,
                                   C=29,
                                   dimension=512,
                                   mask_radius=240,
                                   max_velocity: float = 5.0) -> float:
    """
    Evaluate average velocity directly from images in a time range.

    Uses actual timestamps from image filenames to compute real velocities,
    filters out tracking artifacts, and returns the mean velocity.

    Args:
        images_path: Path to directory containing images
        start_timestamp: Start time in command format (YYYY-MM-DDThh:mm:ss.nnnnnnZ)
        end_timestamp: End time in command format
        crop_*, block_size, C, dimension, mask_radius: Image processing params
        max_velocity: Maximum plausible velocity for filtering artifacts

    Returns:
        Average velocity in pixels/second over the time interval
    """
    # 1. Get images with their datetimes in the specified range
    images_with_datetimes = get_images_with_datetimes(
        images_path, start_timestamp, end_timestamp
    )

    if not images_with_datetimes:
        raise ValueError(f"No images found in time range {start_timestamp} to {end_timestamp}")

    logging.info("Found %d images in time range", len(images_with_datetimes))

    # 2. Get centroids and datetimes (only for images where bot was detected)
    centroids, datetimes = get_centroids_and_datetimes(
        images_with_datetimes,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        block_size=block_size,
        C=C,
        dimension=dimension,
        mask_radius=mask_radius
    )

    if len(centroids) < 2:
        raise ValueError(f"Need at least 2 detected centroids, got {len(centroids)}")

    logging.info("Detected bot in %d/%d images", len(centroids), len(images_with_datetimes))

    # 3. Compute velocities over time using actual timestamps
    velocities, delta_times = compute_velocities_over_time(
        centroids, datetimes, max_velocity=max_velocity
    )

    if not velocities:
        raise ValueError("No valid velocity measurements (all filtered as artifacts)")

    # 4. Return average velocity
    avg_velocity = np.mean(velocities)

    logging.info("Computed %d velocity measurements, average: %.4f pixels/s",
                 len(velocities), avg_velocity)

    return avg_velocity
    

# velocity = evaluate_velocity(
#     "/Users/kam/dev/biosense/data_processing/vis/videos/12_frame_raw_videos_avc1/fm_test_input_000068.mp4",
#     output_path=".",
#     visualize=True
# )
# print(velocity)