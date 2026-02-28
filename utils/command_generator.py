import random
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from .models import (
    TemperatureInstructionModel,
    VibrationInstructionModel,
    ElectricalInstructionModel,
    ChemicalInstructionModel,
    LightInstructionModel,
    LightIntensity,
    CameraInstructionModel,
    EndInstructionModel,
    CommandModel,
    BatchOpType,
    AnyInstructionModel,
    SystemCapabilitiesModel,
)
from .settings import get_settings
from .constants import INTERVENTION_LENGTH_MS

MIN_GENERATED_INSTRUCTIONS = 2
MAX_GENERATED_INSTRUCTIONS = 5

MIN_DURATION_MS = 2000
MAX_DURATION_MS = 10000


def gen_temperature_model(
    settings: SystemCapabilitiesModel,
) -> TemperatureInstructionModel:
    temp_min, temp_max = settings.temperature.temperature_range_c
    temperature_c = random.randrange(
        int(temp_min * 2), int(temp_max * 2) + 1) / 2.0
    return TemperatureInstructionModel(
        temperature_c=temperature_c,
        duration_ms=random.randint(
            settings.temperature.duration_ms.min, settings.temperature.duration_ms.max
        ),
    )


def gen_vibration_model(settings: SystemCapabilitiesModel) -> VibrationInstructionModel:
    frequency_min, frequency_max = settings.vibration.frequency_range_hz
    return VibrationInstructionModel(
        frequency_hz=random.uniform(frequency_min, frequency_max),
        amplitude=random.random() * settings.vibration.amplitude_max,
        duration_ms=random.randint(MIN_DURATION_MS, INTERVENTION_LENGTH_MS),
    )


def gen_electrical_model(
    settings: SystemCapabilitiesModel,
) -> ElectricalInstructionModel:
    return ElectricalInstructionModel(
        current_ma=0.5 * random.randint(1, 41),  # 0.5mA to 20mA
        angle_degrees=22.5 * random.randint(0, 16),
        # frequency_hz=0 if random.randint(0, 1) == 0 else 50,
        frequency_hz=0,  # DC current only
        duration_ms=random.randint(MIN_DURATION_MS, INTERVENTION_LENGTH_MS),
    )


def gen_chem_model(settings: SystemCapabilitiesModel) -> ChemicalInstructionModel:
    chemicals = settings.chemical.chemical_names
    volume_min, volume_max = settings.chemical.volume_ul

    return ChemicalInstructionModel(
        chemical_name=random.choice(chemicals),
        dilution=random.choice(settings.chemical.dilutions_percent),
        volume_ul=random.randint(volume_min, volume_max),
    )


def gen_light_model(settings: SystemCapabilitiesModel) -> LightInstructionModel:
    red_min, red_max = settings.light.red_intensity_step
    green_min, green_max = settings.light.green_intensity_step
    blue_min, blue_max = settings.light.blue_intensity_step

    return LightInstructionModel(
        intensity_step=LightIntensity(
            red=random.randint(red_min, red_max),
            green=random.randint(green_min, green_max),
            blue=random.randint(blue_min, blue_max),
        ),
        duration_ms=random.randint(MIN_DURATION_MS, MAX_DURATION_MS),
    )


def gen_camera_model(settings: SystemCapabilitiesModel) -> CameraInstructionModel:
    return CameraInstructionModel(
        delay_after_capture_ms=random.randint(
            MIN_DURATION_MS, MAX_DURATION_MS),
    )


def gen_end_model(settings: SystemCapabilitiesModel) -> EndInstructionModel:
    return EndInstructionModel()


def get_model_generator(
    cap: BatchOpType,
) -> Callable[[SystemCapabilitiesModel], AnyInstructionModel]:
    if cap == BatchOpType.Temperature:
        return gen_temperature_model
    elif cap == BatchOpType.Vibration:
        return gen_vibration_model
    elif cap == BatchOpType.Electrical:
        return gen_electrical_model
    elif cap == BatchOpType.Chemical:
        return gen_chem_model
    elif cap == BatchOpType.Light:
        return gen_light_model
    elif cap == BatchOpType.CameraHigh:
        return gen_camera_model
    elif cap == BatchOpType.End:
        return gen_end_model
    else:
        raise ValueError(f"Invalid type {cap}")


def generate_command_model(
    batch_id: str,
    cap: BatchOpType,
    n_instructions: int = 1,
    n_runs: int = 1,
    custom_gen_func: Optional[
        Callable[[SystemCapabilitiesModel], AnyInstructionModel]
    ] = None,
) -> CommandModel[AnyInstructionModel]:
    gen_func = custom_gen_func if custom_gen_func else get_model_generator(cap)

    settings = get_settings()
    instructions: list[AnyInstructionModel] = [
        gen_func(settings) for _ in range(n_instructions)
    ]

    return CommandModel(
        type=cap,
        timestamp=datetime.now(timezone.utc),
        batch_id=batch_id,
        command_id=str(uuid.uuid4()),
        runs=n_runs,
        instructions=instructions,
    )
