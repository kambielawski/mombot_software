# from __future__ import annotations

from .batch_models import (
    MappedBatchModel,
    BatchModel,
    BatchLogEntry,
    BatchLogModel
)
from .command_models import (
    CommandInstructionsType,
    CommandFileInfo,
    CommandModel,
)
from .enum_models import (
    RangeVar,
    Directories,
    CommandResult,
    ArenaType,
    BatchOpType,
    StatusOperation,
    CommandEndReason,
    Range,
    LightIntensity,
    BatchId,
    StationId,
)
from .instruction_models import (
    has_duration,
    TemperatureInstructionModel,
    VibrationInstructionModel,
    ElectricalInstructionModel,
    LightInstructionModel,
    ChemicalInstructionModel,
    CameraInstructionModel,
    EndInstructionModel,
    AnyInstructionModel,
    TimedInstructions,
)
from .settings_models import (
    SystemTemperatureSettings,
    SystemVibrationSettings,
    SystemElectricalSettings,
    SystemLightSettings,
    SystemChemicalSettings,
    SystemCameraSettings,
    SystemDebugSettings,
    SystemCapabilitiesModel,
)

__all__ = [
    # region Common Models
    "RangeVar",
    "Directories",
    "CommandResult",
    "ArenaType",
    "BatchOpType",
    "StatusOperation",
    "CommandEndReason",
    "Range",
    "LightIntensity",
    "BatchId",
    "StationId",
    # endregion
    # region Batch Models
    "MappedBatchModel",
    "BatchModel",
    "BatchLogEntry",
    "BatchLogModel",
    # endregion
    # region Instruction Models
    "has_duration",
    "TemperatureInstructionModel",
    "VibrationInstructionModel",
    "ElectricalInstructionModel",
    "LightInstructionModel",
    "ChemicalInstructionModel",
    "CameraInstructionModel",
    "EndInstructionModel",
    "AnyInstructionModel",
    "TimedInstructions",
    # endregion
    # region Command Models
    "CommandInstructionsType",
    "CommandFileInfo",
    "CommandModel",
    # endregion
    # region Settings Models
    "SystemTemperatureSettings",
    "SystemVibrationSettings",
    "SystemElectricalSettings",
    "SystemLightSettings",
    "SystemChemicalSettings",
    "SystemCameraSettings",
    "SystemDebugSettings",
    "SystemCapabilitiesModel",
    # endregion
]


def __dir__() -> list[str]:
    return list(__all__)
