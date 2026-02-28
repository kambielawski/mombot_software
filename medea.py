#!/usr/bin/env python3
"""
Simulates the Medea process.
"""

import os
import sys
import json
import subprocess
import shutil
import time
import logging
import threading
import traceback
import logging
from abc import ABC, abstractmethod

from utils.constants import (
    MOMBOT_ROOT_DIR,
    AVAILABLE_FOLDER_PATH,
    IN_PROGRESS_FOLDER_PATH,
    ARCHIVE_FOLDER_PATH
)

################################################################################


class EventHandler(ABC):
    """
    Abstract class for handling events
    """
    @abstractmethod
    def check_for_events(self):
        pass

    @abstractmethod
    def handle_event(self, event):
        pass

    def watch(self):
        while True:
            events = self.check_for_events()
            if len(events) > 0:
                for event in events:
                    self.handle_event(event)
            time.sleep(0.1)

################################################################################


class FileCreationHandler(EventHandler):
    """
    Abstract class for handling events
    """

    def __init__(self, watch_path, callback=None):
        self.callback = callback
        self.watch_path = watch_path
        self.last_contents = set()

    def check_for_events(self):
        events = []

        if os.path.exists(self.watch_path):
            current_contents = set(os.listdir(self.watch_path))

            # Check for new files
            new_files = current_contents - self.last_contents
            for file in new_files:
                events.append(file)

            self.last_contents = current_contents

        return events

    @abstractmethod
    def handle_event(self, file_name):
        pass

################################################################################


class NewlyAvailableBatchHandler(FileCreationHandler):
    """
    Concrete class for handling the event of a new batch becoming available
    signified by a batch file being uploaded to batches_available_for_medea
    """

    def __init__(self, available_folder_path):
        super().__init__(available_folder_path)
        self.active_batch_ids = set()

    def handle_event(self, file_name):
        """
        Handles a new file being uploaded to batches_available_for_medea
        """
        super().handle_event(file_name)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s')

        if file_name.endswith('.json'):
            batch_dict = self.validate_batch_file(file_name)
            batch_id = batch_dict['batch_id']

            # Check for active batch...
            if batch_id in self.active_batch_ids:
                logging.info("Batch %s is already active, skipping...", batch_id)
                return
            else:
                self.active_batch_ids.add(batch_id)

            # Move batch spec from available folder to in-progress folder
            available_batch_file_path = os.path.join(
                AVAILABLE_FOLDER_PATH, f"batch-{batch_id}.json")
            batch_folder_path = os.path.join(
                f"{IN_PROGRESS_FOLDER_PATH}/batch-{batch_id}", f"batch-{batch_id}.json")

            for _ in range(3):
                try:
                    shutil.move(available_batch_file_path, batch_folder_path)
                    break
                except Exception as e:
                    logging.error("Error moving batch file: %s", e)
                    logging.error("Stack trace:\n%s", traceback.format_exc())
                    time.sleep(0.5)
                    continue

            # Spawn babysitter process
            self.spawn_babysitter_process(batch_id)

    def validate_batch_file(self, file_name):
        """
        Validates a batch file 
        TODO: Move this to some utils folder... 
        """
        for i in range(3):
            try:
                # First try UTF-8
                with open(f"{AVAILABLE_FOLDER_PATH}/{file_name}", 'r', encoding='utf-8') as file:
                    batch_str = file.read()
                    batch_spec = json.loads(batch_str)
                    return batch_spec
            except UnicodeDecodeError:
                # If UTF-8 fails, try reading as UTF-16
                try:
                    with open(f"{AVAILABLE_FOLDER_PATH}/{file_name}", 'r', encoding='utf-16') as file:
                        batch_str = file.read()
                        batch_spec = json.loads(batch_str)
                        return batch_spec
                except Exception as e:
                    # If both fail, try reading in binary mode and decode manually
                    try:
                        with open(f"{AVAILABLE_FOLDER_PATH}/{file_name}", 'rb') as file:
                            batch_str = file.read().decode('utf-8-sig')  # utf-8-sig handles BOM if present
                            batch_spec = json.loads(batch_str)
                            return batch_spec
                    except Exception as e:
                        logging.error("Error validating batch file: %s", e)
                        time.sleep(0.5)
                        continue

        raise IOError("Error validating batch file: %s", file_name)

    def spawn_babysitter_process(self, batch_id: str):
        """
        Spawns a babysitter process for the given batch ID
        """
        logging.info("New batch available: batch-%s", batch_id)

        command = ["python3", "babysitter.py", "--batch_id", batch_id]
        process = subprocess.Popen(command, start_new_session=True)

################################################################################


class NewArchiveHandler(FileCreationHandler):
    """
    Detects and responds to batches being moved into the archive folder
    """

    def handle_event(self, item):
        super().handle_event(item)

        # Check if the new item is a directory
        if os.path.isdir(f'{ARCHIVE_FOLDER_PATH}/{item}'):
            logging.info("New batch completed: %s", item)

            input()

            logging.info("Running specifications generator on video data...")

            # python3 ./biosense_SR/specifications.py <BATCH FOLDER PATH> <DESTINATION FOLDER>
            os.system(
                f'python3 "./biosense_specifications/specifications.py" "{ARCHIVE_FOLDER_PATH}/{item}/videos" "{ARCHIVE_FOLDER_PATH}/{item}/analysis"')

            input()

            logging.info("Running success signature analysis on video data...")
            # python3 ./biosense_success_signatures/success_signatures.py <BATCH FOLDER PATH> <DESTINATION FOLDER>
            os.system(
                f'python3 "./biosense_success_signatures/success_signatures.py" "{ARCHIVE_FOLDER_PATH}/{item}/videos" "{ARCHIVE_FOLDER_PATH}/{item}/analysis"')

            input()

################################################################################


class Medea:
    """
    Class for running the Medea process
    """

    def __init__(self):
        self.threads = []

    def run(self):
        for event_handler in [
            NewlyAvailableBatchHandler(AVAILABLE_FOLDER_PATH),
            # NewArchiveHandler(ARCHIVE_FOLDER_PATH)
        ]:
            thread = threading.Thread(target=event_handler.watch)
            thread.start()
            self.threads.append(thread)

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")


def launch_ui():
    print("\n" + "="*60)
    print("  MEDEA LAUNCH")
    print("="*60)
    print(f"\n  MOMBOT_ROOT_DIR:\n    {MOMBOT_ROOT_DIR}\n")
    print(f"  AVAILABLE_FOLDER contents ({AVAILABLE_FOLDER_PATH}):")
    if os.path.exists(AVAILABLE_FOLDER_PATH):
        items = os.listdir(AVAILABLE_FOLDER_PATH)
        if items:
            for item in sorted(items):
                print(f"    - {item}")
        else:
            print("    (empty)")
    else:
        print("    (folder does not exist)")
    print("\n" + "="*60)
    confirm = input("  Launch Medea? [y/N]: ").strip().lower()
    return confirm == 'y'


if __name__ == "__main__":
    if not launch_ui():
        print("Aborted.")
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{MOMBOT_ROOT_DIR}/medea.log')
        ],
    )
    try:
        logging.info("Running")
        logging.info("Loaded into directory: %s", MOMBOT_ROOT_DIR)

        medea = Medea()
        medea.run()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        os.system("./scripts/reset_medea.sh")
        sys.exit(0)
