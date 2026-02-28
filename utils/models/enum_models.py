from enum import StrEnum
from typing import Annotated, NamedTuple, Generic, TypeVar

from pydantic import StringConstraints

RangeVar = TypeVar("RangeVar", int, float)


class Directories(StrEnum):
    Available = "batches_available_for_medea"
    Pending = "pending"
    Data = "data"


class CommandResult(StrEnum):
    Cancelled = "cancelled"
    Progress = "progress"
    Success = "success"
    Failure = "failure"


class ArenaType(StrEnum):
    Growth = "growth"
    Test = "test"


class BatchOpType(StrEnum):
    Temperature = "temperature"
    Vibration = "vibration"
    Electrical = "electrical"
    Chemical = "chemical"
    Light = "light"
    CameraHigh = "camera_high"
    Move = "move"
    End = "end"


class StatusOperation(StrEnum):
    CommandStart = "command_start"
    CommandEnd = "command_end"
    ImageCapture = "image_capture"


class CommandEndReason(StrEnum):
    Complete = "completed"
    Interrupted = "interrupted"
    Superseded = "superseded"
    ThreadOperationError = "thread_operation_error"
    InvalidBatch = "invalid_batch"
    InvalidInstruction = "invalid_instruction"
    OutOfRange = "out_of_range"
    ParseFailure = "parse_failure"
    CapabilityOffline = "capability_offline"
    ChemStimRunning = "chem_stim_running"
    CoordinatorNotFound = "coordinator_not_found"
    CoordinatorRejectedGoal = "coordinator_rejected_goal"
    ServiceGoalFailed = "service_goal_failed"
    ChemStimGoalFailed = "chem_stim_goal_failed"
    StationUnavailable = "station_unavailable"
    CoordinatorFault = "coordinator_fault"
    BatchEnded = "batch_ended"


class Range(NamedTuple, Generic[RangeVar]):
    min: RangeVar
    max: RangeVar


class LightIntensity(NamedTuple):
    red: int
    green: int
    blue: int


BatchId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"\d{6}"),
]
StationId = Annotated[
    str,
    StringConstraints(to_lower=True, strip_whitespace=True, pattern="[0-9a-fA-F]{4}"),
]
