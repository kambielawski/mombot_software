"""
Class for babysitting a batch
"""

import os
import sys
import argparse
import uuid
import json
import threading
import traceback
import logging
import subprocess
from datetime import datetime, timezone, timedelta
import time
import re

from medea import FileCreationHandler
from utils.image_processing import (
    get_timestamp
)

from utils.models.command_models import CommandModel, BatchOpType
from utils.models.instruction_models import CameraInstructionModel
from utils.medea_models import InterventionModel

from utils.constants import (
    IN_PROGRESS_FOLDER_PATH,
    INITIAL_VIDEO_DURATION_MS,
    COMMANDS_FOLDER_PATH,
    DP_FM_PATH,
    MOMBOT_ROOT_DIR,
    NN_FM_PATH,
    MIN_CAMERA_DELAY_MS,
    POST_INTERVENTION_OBSERVATION_PERIOD_MS
)
from video_processing.evaluate_velocity import evaluate_velocity_from_images

# !!!!!! Mutable global state !!!!!!!!
GLOBAL_COMMAND_COUNTER = 1
GLOBAL_INTERVENTION_COUNTER = 1

# Command line arguments (global state)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_id", type=str, required=True)
args = parser.parse_args()
batch_id = args.batch_id

################################################################################


class CommandExecutor:
    """
    Class for sending commands to mombot
    """

    def __init__(self):
        pass

    def request_process(self, command_dict):
        global GLOBAL_COMMAND_COUNTER
        command_dict['command_generated_timestamp'] = time.time()
        command_dict['command_id'] = GLOBAL_COMMAND_COUNTER
        GLOBAL_COMMAND_COUNTER += 1

        self.send_command_to_mombot(command_dict)

    def request_video(self, duration_ms):
        # Generates random camera command
        camera_instructions = [
            CameraInstructionModel(delay_after_capture_ms=MIN_CAMERA_DELAY_MS)
        ]
        command_model = CommandModel(
            type=BatchOpType.CameraHigh,
            timestamp=datetime.now(timezone.utc),
            batch_id=batch_id,
            command_id=str(uuid.uuid4()),
            runs=duration_ms // MIN_CAMERA_DELAY_MS,
            instructions=camera_instructions,
        )
        self.send_command_to_mombot(command_model)

    def request_dispose(self):
        dispose_command = {
            "command_id": GLOBAL_COMMAND_COUNTER,
            "batch_id": batch_id,
            "process": "dispose",
            "instructions": []
        }
        self.send_command_to_mombot(dispose_command)

    def send_command_to_mombot(self, command_model: CommandModel):
        command_json = command_model.model_dump_json(indent=2)
        logging.info("Sending %s command to mombot", command_model.type)
        with open(f"{COMMANDS_FOLDER_PATH}/batch-{batch_id}_{command_model.type}.json", "w") as file:
            file.write(command_json)

################################################################################


class VideoFileHandler(FileCreationHandler):
    def __init__(self, watch_path, train_callback=None, inverse_callback=None):
        super().__init__(watch_path, inverse_callback)
        self.train_callback = train_callback
        self.inverse_callback = inverse_callback

    def handle_event(self, video_file_name):
        super().handle_event(video_file_name)

        if video_file_name.endswith('.mp4'):
            logging.info("Video file detected: %s", video_file_name)
            # self.train_callback(video_file_path=video_file_name, n_video_files=1)
            # Inverse should generate an intervention file
            self.inverse_callback(
                video_file_path=f'{self.watch_path}/{video_file_name}', objective=None)

################################################################################


class ImageFileHandler(FileCreationHandler):
    # TODO: Do we need this?
    def __init__(self, watch_path):
        super().__init__(watch_path)

    def handle_event(self, image_file_name):
        super().handle_event(image_file_name)

        if image_file_name.endswith('.png'):
            logging.info("Image file detected: %s", image_file_name)

################################################################################


class InterventionFileHandler(FileCreationHandler):
    """
    Class for handling a new intervention file
    """

    def __init__(self, watch_path, callback=None):
        super().__init__(watch_path, callback)
        self.command_executor = CommandExecutor()
        self.intervention_file_watcher = None

    def handle_event(self, file_name):
        super().handle_event(file_name)

        # Ensure the new item is an intervention folder
        if not re.fullmatch(r"intervention-\d{6}", file_name):
            return

        folder_name = file_name
        intervention_folder_path = f"{self.watch_path}/{folder_name}"

        self.intervention_file_watcher = threading.Thread(
            target=self.watch_intervention, args=(intervention_folder_path,), daemon=True)
        self.intervention_file_watcher.start()

    def watch_intervention(self, intervention_folder_path):
        intervention_id = intervention_folder_path.split('-')[-1]
        intervention_file = f"{intervention_folder_path}/intervention-{intervention_id}.json"

        logging.info("Waiting for intervention-%s", intervention_id)
        logging.info("Watching for %s", intervention_file)

        while True:
            try:
                if os.path.exists(intervention_file):
                    with open(intervention_file, "r") as file:
                        intervention = json.load(file)

                    intervention_model = InterventionModel(batch_id=intervention['batch_id'],
                                                           intervention_id=intervention['intervention_id'],
                                                           commands=[
                                                               CommandModel(
                                                                   timestamp=datetime.now(
                                                                       timezone.utc),
                                                                   batch_id=batch_id,
                                                                   command_id=command['command_id'],
                                                                   runs=command['runs'],
                                                                   instructions=command['instructions'],
                                                                   start_time=None,
                                                                   end_time=None,
                                                                   end_reason=None,
                                                                   type=command['type']
                                                               ) for command in intervention['commands']])

                    self.execute_intervention(intervention_model)
                    return
            except Exception as e:
                logging.error("Error: %s", e)
                logging.error("Stack trace: %s", traceback.format_exc())
            time.sleep(0.5)

    def execute_intervention(self, intervention: InterventionModel):
        logging.info("Executing intervention %s", intervention.intervention_id)
        for command in intervention.commands:
            self.command_executor.send_command_to_mombot(command)

################################################################################


class InterventionVideoHandler(FileCreationHandler):
    def __init__(self, watch_path, images_path, videos_path, interventions_path, callback=None):
        super().__init__(watch_path, callback)
        self.watch_path = watch_path
        self.images_path = images_path
        self.videos_path = videos_path
        self.interventions_path = interventions_path
        self.watcher_thread = None

    def handle_event(self, file_name):
        super().handle_event(file_name)

        if "electrical" in file_name:
            logging.info("Electrical command detected: %s", file_name)
            self.watcher_thread = threading.Thread(
                target=self.watch_json, args=(file_name,), daemon=True)
            self.watcher_thread.start()

    def watch_json(self, file_name):
        json_path = f"{self.watch_path}/{file_name}"
        logging.info("[batch_%s babysitter] Watching electrical command: %s", batch_id, file_name)

        # Wait for electrical stim to end
        while True:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if "end_time" in data and data["end_time"]:
                        stim_end_time = data["end_time"]
                        estim_duration = data["instructions"][0]["duration_ms"]
                        logging.info("Electrical stim ended at %s, waiting %d ms for post-observation",
                                     stim_end_time, POST_INTERVENTION_OBSERVATION_PERIOD_MS)
                        break
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            time.sleep(0.5)

        # Wait for post-observation period
        time.sleep(POST_INTERVENTION_OBSERVATION_PERIOD_MS / 1000.0)

        # Compute observation end timestamp
        stim_end_dt = datetime.fromisoformat(stim_end_time.replace('Z', '+00:00'))
        obs_end_dt = stim_end_dt + timedelta(milliseconds=POST_INTERVENTION_OBSERVATION_PERIOD_MS)
        obs_end_time = obs_end_dt.isoformat().replace('+00:00', 'Z')

        # Extract velocity directly from images (no video creation)
        try:
            post_velocity = evaluate_velocity_from_images(
                images_path=self.images_path,
                start_timestamp=stim_end_time,
                end_timestamp=obs_end_time
            )
            logging.info("Post-intervention velocity: %s pixels/s", post_velocity)
        except Exception as e:
            logging.error("Error extracting post-intervention velocity: %s", e)
            return

        # Get pre_velocity from intervention file
        pre_velocity = self._get_pre_velocity_from_intervention()
        if pre_velocity is None:
            logging.error("Could not find pre_velocity from intervention file")
            return

        # Update nonlinear_fm_cache.json
        self._update_fm_cache(pre_velocity, estim_duration, post_velocity)

    def _get_pre_velocity_from_intervention(self):
        """Find the most recent intervention and return its pre_velocity."""
        try:
            intervention_folders = sorted(
                [d for d in os.listdir(self.interventions_path) if d.startswith("intervention-")],
                reverse=True
            )
            if not intervention_folders:
                return None
            latest = intervention_folders[0]
            intervention_file = f"{self.interventions_path}/{latest}/{latest}.json"
            with open(intervention_file, "r") as f:
                intervention = json.load(f)
            return intervention.get("pre_velocity")
        except Exception as e:
            logging.error("Error reading intervention file: %s", e)
            return None

    def _update_fm_cache(self, pre_velocity, estim_duration, post_velocity):
        """Update nonlinear_fm_cache.json with new observation."""
        cache_path = "nonlinear_fm_cache.json"
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

        cache[batch_id] = [pre_velocity, estim_duration, post_velocity]

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=4)
        logging.info("Updated FM cache: batch %s -> [%s, %s, %s]", batch_id, pre_velocity, estim_duration, post_velocity)

    def assemble_video(self):
        # Get all png files and sort them
        png_files = [f for f in os.listdir(
            self.images_path) if f.endswith('.jpg') or f.endswith('.png')]
        png_files.sort()

        logging.info("Assembling video from %d images", len(png_files))

        try:
            chunks = sort_images(self.images_path)
            for chunk in chunks:
                assemble_video(self.images_path, self.videos_path, chunk)
        except Exception as e:
            logging.error("Error assembling video: %s", e)
            logging.error("Stack trace: %s", e.__traceback__)

        logging.info("Created video from %d images: %s", len(png_files), self.videos_path)


################################################################################

class InitialVideoHandler(FileCreationHandler):
    def __init__(self, images_path, batch_folder_path, inverse_callback):
        self.images_path = images_path
        self.batch_folder_path = batch_folder_path
        self.inverse_callback = inverse_callback
        self.velocity_computed = False
        super().__init__(images_path)

    def handle_event(self, file_name):
        super().handle_event(file_name)

        if self.velocity_computed:
            return

        time_diff_ms = self.total_time_diff_ms()
        if time_diff_ms >= INITIAL_VIDEO_DURATION_MS:
            logging.info("Time difference (%s ms) exceeds INITIAL_VIDEO_DURATION_MS, computing pre-velocity", time_diff_ms)
            self.compute_and_save_velocity()
            self.velocity_computed = True

    def total_time_diff_ms(self):
        # Get all jpg/png files
        image_files = [f for f in os.listdir(
            self.images_path) if f.endswith('.jpg') or f.endswith('.png')]
        if not image_files:
            return 0

        image_files.sort(key=get_timestamp)

        # Check time difference between first and last image
        first_timestamp = get_timestamp(image_files[0])
        last_timestamp = get_timestamp(image_files[-1])
        time_diff_ms = (last_timestamp -
                        first_timestamp).total_seconds() * 1000

        return time_diff_ms

    def compute_and_save_velocity(self):
        try:
            # Get all image files sorted by timestamp
            image_files = [f for f in os.listdir(
                self.images_path) if f.endswith('.jpg') or f.endswith('.png')]
            image_files.sort(key=get_timestamp)

            # Get start and end timestamps
            first_timestamp = get_timestamp(image_files[0])
            last_timestamp = get_timestamp(image_files[-1])

            # Convert datetime to string format expected by evaluate_velocity_from_images
            # (YYYY-MM-DDThh:mm:ss.nnnnnnZ format with colons)
            start_timestamp_str = first_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'
            end_timestamp_str = last_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'

            logging.info("Computing velocity from images between %s and %s", start_timestamp_str, end_timestamp_str)

            # Evaluate velocity directly from images
            pre_velocity = evaluate_velocity_from_images(
                images_path=self.images_path,
                start_timestamp=start_timestamp_str,
                end_timestamp=end_timestamp_str
            )

            logging.info("Pre-intervention velocity: %s pixels/s", pre_velocity)

            # Write pre_velocity to text file in batch folder
            pre_velocity_file_path = f"{self.batch_folder_path}/pre_velocity.txt"
            with open(pre_velocity_file_path, 'w') as f:
                f.write(str(pre_velocity))

            logging.info("Wrote pre_velocity to %s", pre_velocity_file_path)

            # Trigger inverse with the pre_velocity file path
            self.inverse_callback(pre_velocity_file_path=pre_velocity_file_path)

        except Exception as e:
            logging.error("Error computing velocity: %s", e)
            logging.error("Stack trace: %s", traceback.format_exc())

################################################################################


class BatchBabysitter:
    """
    Class for babysitting a batch / batch-specific logic
    """

    def __init__(self):
        logging.info("Initializing babysitter for batch-%s", batch_id)
        self.batch_folder_path = f"{IN_PROGRESS_FOLDER_PATH}/batch-{batch_id}"
        self.intervention_folder_path = f"{self.batch_folder_path}/interventions"
        self.images_folder_path = f"{self.batch_folder_path}/images"
        self.videos_folder_path = f"{self.batch_folder_path}/videos"
        self.weights_folder_path = f"{self.batch_folder_path}/weights"
        self.analysis_folder_path = f"{self.batch_folder_path}/analysis"
        self.commands_folder_path = f"{self.batch_folder_path}/commands"
        self.logs_folder_path = f"{self.batch_folder_path}/logs"
        self.command_executor = CommandExecutor()
        self.image_file_handler = ImageFileHandler(self.images_folder_path)
        self.video_file_handler = VideoFileHandler(
            self.videos_folder_path, train_callback=self.run_train, inverse_callback=self.run_inverse)
        self.intervention_file_handler = InterventionFileHandler(
            self.intervention_folder_path)
        # self.camera_command_handler = CameraCommandHandler(self.commands_folder_path, images_path=self.images_folder_path, videos_path=self.videos_folder_path)
        self.initial_video_handler = InitialVideoHandler(images_path=self.images_folder_path,
                                                         batch_folder_path=self.batch_folder_path,
                                                         inverse_callback=self.run_inverse)
        self.intervention_video_handler = InterventionVideoHandler(watch_path=self.commands_folder_path,
                                                                   images_path=self.images_folder_path,
                                                                   videos_path=self.videos_folder_path,
                                                                   interventions_path=self.intervention_folder_path)

        self.startup()

    def startup(self):
        os.makedirs(self.images_folder_path, exist_ok=True)
        os.makedirs(self.intervention_folder_path, exist_ok=True)
        os.makedirs(self.videos_folder_path, exist_ok=True)
        os.makedirs(self.analysis_folder_path, exist_ok=True)
        os.makedirs(self.logs_folder_path, exist_ok=True)

        # Request initial video
        logging.info("Requesting initial video")
        # 2^31 - 1, max 32-bit signed integer
        self.command_executor.request_video(duration_ms=2147483647)

    def watch(self):
        image_handler_thread = threading.Thread(
            target=self.image_file_handler.watch, daemon=True)
        image_handler_thread.start()

        # video_handler_thread = threading.Thread(
        #     target=self.video_file_handler.watch, daemon=True)
        # video_handler_thread.start()

        intervention_handler_thread = threading.Thread(
            target=self.intervention_file_handler.watch, daemon=True)
        intervention_handler_thread.start()

        # camera_command_handler_thread = threading.Thread(target=self.camera_command_handler.watch, daemon=True)
        # camera_command_handler_thread.start()

        initial_video_handler_thread = threading.Thread(
            target=self.initial_video_handler.watch, daemon=True)
        initial_video_handler_thread.start()

        intervention_video_handler_thread = threading.Thread(
            target=self.intervention_video_handler.watch, daemon=True)
        intervention_video_handler_thread.start()

        while True:
            time.sleep(0.5)

    def intervention_watcher(self):
        """
        Watches for the end of an intervention by watching the end_timestamp
        of the associated camera command.
        """
        pass

    def run_train(self, video_file_path, n_video_files):
        # TODO: Match interventions to video files...

        self.weights_folder_path = "/tmp"

        logging.info("Training differentiable physics (DP) forward model")

        subprocess.Popen(['python3',
                          f'./{DP_FM_PATH}/train_fm.py',
                          self.videos_folder_path,
                          self.weights_folder_path])

        logging.info("Training neural network (NN) forward model")

        subprocess.Popen(['python3',
                          f'./{NN_FM_PATH}/train_fm1.py',
                          self.videos_folder_path,
                          self.weights_folder_path])

    def run_inverse(self, pre_velocity_file_path, objective=None):
        global GLOBAL_INTERVENTION_COUNTER

        # TODO: Dead batch detection before this

        # TODO: Inverse method should be interchangeable

        logging.info("Starting inverse method")

        os.system('python3 ./inverse.py "{0}" "{1}" "{2}"'.format(
            self.batch_folder_path, pre_velocity_file_path, GLOBAL_INTERVENTION_COUNTER))

        GLOBAL_INTERVENTION_COUNTER += 1

################################################################################


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format=f'%(asctime)s [%(levelname)s] [%(module)s-{batch_id}] %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout),
                            logging.FileHandler(f'{MOMBOT_ROOT_DIR}/data/batch-{batch_id}/batch_{batch_id}_babysitter.log')
                        ])

    try:
        babysitter = BatchBabysitter()
        babysitter.watch()
    except Exception as e:
        logging.error("Couldn't watch batch %d: %s", batch_id, e)
        logging.error("Stack trace:\n%s", traceback.format_exc())
        exit(1)
