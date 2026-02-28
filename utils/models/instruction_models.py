from typing import Optional, overload, Literal, Protocol, TypeVar, TypeGuard, Annotated

from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo

from .enum_models import Range, RangeVar, BatchId, LightIntensity
from .settings_models import SystemCapabilitiesModel
from ..settings import get_settings


class HasDurationProtocol(Protocol):
    duration_ms: int


T = TypeVar("T")


def has_duration(obj: T) -> TypeGuard[HasDurationProtocol]:
    return hasattr(obj, "duration_ms")


ClampedAngle = Annotated[float, Field(ge=0.0, lt=360.0)]


def validate_settings_context(info: ValidationInfo) -> SystemCapabilitiesModel:
    context = info.context
    settings: SystemCapabilitiesModel = (
        get_settings() if context is None else context.get("settings")
    )

    if settings is None:
        raise ValueError(f"Validator is missing settings context")
    return settings


@overload
def validate_range(
    allowed_range: Range[RangeVar],
    val: Optional[RangeVar],
    raise_on_empty: Literal[True] = True,
) -> RangeVar: ...
@overload
def validate_range(
    allowed_range: Range[RangeVar],
    val: Optional[RangeVar],
    raise_on_empty: Literal[False],
) -> Optional[RangeVar]: ...
def validate_range(
    allowed_range: Range[RangeVar], val: Optional[RangeVar], raise_on_empty: bool = True
) -> Optional[RangeVar]:
    if val is None:
        if not raise_on_empty:
            return None
        raise ValueError("Range value is None, but is required")
    if val < allowed_range.min or val > allowed_range.max:
        raise ValueError(
            f"Value {val} must be between {allowed_range.min} and {allowed_range.max}"
        )
    return val


class TemperatureInstructionModel(BaseModel):
    temperature_c: float
    duration_ms: Optional[int] = None

    @field_validator("temperature_c")
    def validate_temperature_c(cls, val: float, info: ValidationInfo) -> float:
        settings = validate_settings_context(info)
        allowed_range: Range[float] = settings.temperature.temperature_range_c
        return validate_range(allowed_range, val, raise_on_empty=True)

    @field_validator("duration_ms")
    def validate_duration_ms(
        cls, val: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.temperature.duration_ms
        return validate_range(allowed_range, val, raise_on_empty=False)


class VibrationInstructionModel(BaseModel):
    frequency_hz: float
    amplitude: float
    duration_ms: Optional[int] = None

    @field_validator("frequency_hz")
    def validate_frequency_hz(cls, val: float, info: ValidationInfo) -> float:
        settings = validate_settings_context(info)
        allowed_range = settings.vibration.frequency_range_hz
        return validate_range(allowed_range, val)

    @field_validator("amplitude")
    def validate_amplitude(cls, val: float, info: ValidationInfo) -> float:
        settings = validate_settings_context(info)
        if val > settings.vibration.amplitude_max:
            raise ValueError(
                f"Amplitude value {val} must be <= {settings.vibration.amplitude_max}"
            )
        return val

    @field_validator("duration_ms")
    def validate_duration_ms(
        cls, val: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.vibration.duration_ms
        return validate_range(allowed_range, val, raise_on_empty=False)


class ElectricalInstructionModel(BaseModel):
    current_ma: float
    angle_degrees: ClampedAngle
    frequency_hz: int
    duration_ms: Optional[int] = None

    @field_validator("current_ma")
    def validate_current_ma(cls, val: float, info: ValidationInfo) -> float:
        settings = validate_settings_context(info)
        allowed_range = settings.electrical.current_range_ma
        return validate_range(allowed_range, val)

    @field_validator("frequency_hz")
    def validate_frequency_hz(cls, val: int, info: ValidationInfo) -> int:
        settings = validate_settings_context(info)
        allowed_range = settings.electrical.frequency_range_hz
        return validate_range(allowed_range, val)

    @field_validator("duration_ms")
    def validate_duration_ms(
        cls, val: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.electrical.duration_ms
        return validate_range(allowed_range, val, raise_on_empty=False)


class LightInstructionModel(BaseModel):
    intensity_step: LightIntensity
    duration_ms: Optional[int] = None

    @field_validator("intensity_step")
    def validate_intensity_step(
        cls, val: LightIntensity, info: ValidationInfo
    ) -> LightIntensity:
        settings = validate_settings_context(info)
        validate_range(settings.light.red_intensity_step, val.red)
        validate_range(settings.light.green_intensity_step, val.green)
        validate_range(settings.light.blue_intensity_step, val.blue)
        return val

    @field_validator("duration_ms")
    def validate_duration_ms(
        cls, val: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.light.duration_ms
        return validate_range(allowed_range, val, raise_on_empty=False)


# noinspection PyMethodParameters
class ChemicalInstructionModel(BaseModel):
    chemical_name: str
    dilution: float
    volume_ul: int

    @field_validator("chemical_name")
    def validate_chemical_name(cls, val: str, info: ValidationInfo) -> str:
        settings = validate_settings_context(info)
        if val not in settings.chemical.chemical_names:
            raise ValueError(
                f"Chemical {val} is not in the list of accepted values: {str(settings.chemical.chemical_names)}"
            )
        return val

    @field_validator("dilution")
    def validate_dilution(cls, val: float, info: ValidationInfo) -> Optional[float]:
        settings = validate_settings_context(info)
        allowed_values = settings.chemical.dilutions_percent
        if not val in allowed_values:
            raise ValueError(
                f"Dilution {val} is not an accepted value in: {str(allowed_values)}"
            )
        return val

    @field_validator("volume_ul")
    def validate_volume_ul(cls, val: int, info: ValidationInfo) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.chemical.volume_ul
        return validate_range(allowed_range, val)


class CameraInstructionModel(BaseModel):
    delay_after_capture_ms: int

    @field_validator("delay_after_capture_ms")
    def validate_delay_after_capture_ms(
        cls, val: int, info: ValidationInfo
    ) -> Optional[int]:
        settings = validate_settings_context(info)
        allowed_range = settings.camera.delay_after_capture_ms
        return validate_range(allowed_range, val)


class MoveInstructionModel(BaseModel):
    bot_kernel: Optional[str] = None
    batch_destination: BatchId


class EndInstructionModel(BaseModel):
    # End instruction has no identifying information, everything is pulled from the filename.
    # Use a magic string to hint to the parser
    special_type: Literal["batch_end"] = "batch_end"


AnyInstructionModel = (
    TemperatureInstructionModel
    | VibrationInstructionModel
    | ElectricalInstructionModel
    | LightInstructionModel
    | ChemicalInstructionModel
    | CameraInstructionModel
    | MoveInstructionModel
    | EndInstructionModel
)

TimedInstructions = (
    TemperatureInstructionModel,
    VibrationInstructionModel,
    ElectricalInstructionModel,
    LightInstructionModel,
)
