from datetime import datetime
from pydantic import BaseModel

from .enum_models import Range


class SystemTemperatureSettings(BaseModel):
    temperature_range_c: Range[float]
    temperature_step_c: float
    slew_rate_cpmin: float
    duration_ms: Range[int]
    runs_range: Range[int]

    __hash__ = object.__hash__


class SystemVibrationSettings(BaseModel):
    frequency_range_hz: Range[float]
    frequency_step_hz: float
    amplitude_max: float
    amplitude_step: float
    duration_ms: Range[int]
    runs_range: Range[int]

    __hash__ = object.__hash__


class SystemElectricalSettings(BaseModel):
    current_range_ma: Range[float]
    current_step_ma: float
    frequency_range_hz: Range[int]
    frequency_step_hz: int
    pole_pairs: int
    pole_pair_layout_degrees: dict[str, Range[int]]
    duration_ms: Range[int]
    runs_range: Range[int]

    __hash__ = object.__hash__


class SystemLightSettings(BaseModel):
    red_intensity_step: Range[int]
    blue_intensity_step: Range[int]
    green_intensity_step: Range[int]
    step_intensity: int
    duration_ms: Range[int]
    runs_range: Range[int]

    __hash__ = object.__hash__


class SystemChemicalSettings(BaseModel):
    chemical_names: list[str]
    dilutions_percent: list[float]
    volume_ul: Range[int]
    step_ul: int

    __hash__ = object.__hash__


class SystemCameraSettings(BaseModel):
    resolution_mp: int
    delay_after_capture_ms: Range[int]
    repeat_range: Range[int]

    __hash__ = object.__hash__


class SystemDebugSettings(BaseModel):
    debug_camera_sim_ms: int = 3000

    __hash__ = object.__hash__


class SystemCapabilitiesModel(BaseModel):
    timestamp: datetime
    temperature: SystemTemperatureSettings
    vibration: SystemVibrationSettings
    electrical: SystemElectricalSettings
    light: SystemLightSettings
    chemical: SystemChemicalSettings
    camera: SystemCameraSettings
    debug: SystemDebugSettings

    __hash__ = object.__hash__
