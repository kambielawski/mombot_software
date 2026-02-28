from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .constants import MOMBOT_ROOT_DIR

from .models.enum_models import Range
from .models.settings_models import (
    SystemCapabilitiesModel,
    SystemLightSettings,
    SystemTemperatureSettings,
    SystemVibrationSettings,
    SystemElectricalSettings,
    SystemChemicalSettings,
    SystemCameraSettings,
    SystemDebugSettings,
)
DEFAULT_MAX_DURATION = (2**63) - 1


def create_default_settings() -> SystemCapabilitiesModel:

    temperature_settings = SystemTemperatureSettings(
        temperature_range_c=Range(min=4.0, max=26.0),
        temperature_step_c=0.5,
        slew_rate_cpmin=0.5,
        duration_ms=Range(min=30000, max=DEFAULT_MAX_DURATION),
        runs_range=Range(min=1, max=2**16),
    )

    vibration_settings = SystemVibrationSettings(
        frequency_range_hz=Range(min=7, max=200),
        frequency_step_hz=0.5,
        amplitude_max=1.0,
        amplitude_step=0.01,
        duration_ms=Range(min=1000, max=DEFAULT_MAX_DURATION),
        runs_range=Range(min=1, max=2**16),
    )

    electrical_settings = SystemElectricalSettings(
        current_range_ma=Range(min=0.0, max=20.0),
        current_step_ma=0.5,
        frequency_range_hz=Range(min=0.0, max=50.0),
        frequency_step_hz=1,
        pole_pairs=4,
        pole_pair_layout_degrees={
            "pair-1": Range(min=0, max=180),
            "pair-2": Range(min=45, max=225),
            "pair-3": Range(min=90, max=270),
            "pair-4": Range(min=135, max=315),
        },
        duration_ms=Range(min=1000, max=DEFAULT_MAX_DURATION),
        runs_range=Range(min=1, max=2**16),
    )

    light_settings = SystemLightSettings(
        red_intensity_step=Range(min=0, max=256),
        green_intensity_step=Range(min=0, max=256),
        blue_intensity_step=Range(min=0, max=256),
        step_intensity=1,
        duration_ms=Range(min=100, max=DEFAULT_MAX_DURATION),
        runs_range=Range(min=1, max=2**16),
    )

    chemical_settings = SystemChemicalSettings(
        chemical_names=["D1", "D2", "D3", "D4", "D5"],
        dilutions_percent=[5.0, 25.0, 50.0],
        volume_ul=Range(min=1, max=10),
        step_ul=1,
    )

    camera_settings = SystemCameraSettings(
        resolution_mp=64,
        delay_after_capture_ms=Range(min=1000, max=3600000),
        repeat_range=Range(min=1, max=2**16),
    )

    debug_settings = SystemDebugSettings(debug_camera_sim_ms=3000)

    system_settings = SystemCapabilitiesModel(
        timestamp=datetime.now(timezone.utc),
        temperature=temperature_settings,
        vibration=vibration_settings,
        electrical=electrical_settings,
        light=light_settings,
        chemical=chemical_settings,
        camera=camera_settings,
        debug=debug_settings,
    )

    return system_settings


def get_root_dir() -> Path:
    root_dir_str: Optional[str] = MOMBOT_ROOT_DIR
    root_path = Path(root_dir_str)
    if root_path.is_file():
        raise FileExistsError("Mombot root is not a directory")
    root_path.mkdir(exist_ok=True)
    if not root_path.is_dir() or not root_path.exists():
        raise FileNotFoundError(
            "Root path does not exist, or is not a directory")
    return root_path


def get_settings() -> SystemCapabilitiesModel:
    root_dir = get_root_dir()
    file = root_dir / "Mombot_capability.json"
    loaded_settings = create_default_settings()
    if file.exists():
        file_text = file.read_text(encoding="utf-8")
        loaded_settings = SystemCapabilitiesModel.model_validate_json(
            file_text)
    else:
        file.write_text(loaded_settings.model_dump_json(
            indent=2), encoding="utf-8")
    return loaded_settings


if __name__ == "__main__":
    settings = get_settings()
    print(repr(settings))
