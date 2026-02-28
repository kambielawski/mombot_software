from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, RootModel

from .enum_models import (
    StationId,
    BatchId,
    ArenaType,
    BatchOpType,
    StatusOperation,
    CommandEndReason,
)


class MappedBatchModel(BaseModel):
    batch_id: BatchId
    type: int
    well_idx: int
    arena_type: Optional[ArenaType] = None


class BatchModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    timestamp: datetime
    batch_id: BatchId
    station_id: StationId
    plate_type: int
    arena_type: Optional[ArenaType] = None
    capability: list[BatchOpType]
    image: Optional[str]
    description: Optional[str]

    def to_mapped_model(self, well: int) -> MappedBatchModel:
        return MappedBatchModel(
            batch_id=self.batch_id,
            type=self.plate_type,
            well_idx=well,
            arena_type=self.arena_type,
        )


class BatchLogEntry(BaseModel, extra="allow"):
    timestamp: datetime = datetime.now(timezone.utc)

    @staticmethod
    def error(command_id: str, cap: BatchOpType, msg: str) -> BatchLogEntry:
        return BatchLogEntry(command_id=command_id, capability=cap, err_msg=msg)

    @staticmethod
    def command_start(command_id: str, cap: BatchOpType) -> BatchLogEntry:
        return BatchLogEntry(
            operation=StatusOperation.CommandStart,
            command_id=command_id,
            capability=cap,
        )

    @staticmethod
    def command_end(
        command_id: str, cap: BatchOpType, end_reason: CommandEndReason
    ) -> BatchLogEntry:
        return BatchLogEntry(
            operation=StatusOperation.CommandEnd,
            command_id=command_id,
            capability=cap,
            end_reason=end_reason,
        )

    @staticmethod
    def command_operation(
        command_id: str,
        cap: BatchOpType,
        operation: StatusOperation,
        **kwargs: Any,
    ) -> BatchLogEntry:

        return BatchLogEntry(
            operation=operation,
            command_id=command_id,
            capability=cap,
            **kwargs,
        )


BatchLogModel = RootModel[list[BatchLogEntry]]
