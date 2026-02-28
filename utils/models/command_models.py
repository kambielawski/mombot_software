from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar, Generic, Optional

from pydantic import ValidationError, BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo

from .enum_models import (
    BatchId,
    BatchOpType,
    CommandEndReason,
    Directories,
)
from .instruction_models import AnyInstructionModel, TimedInstructions
from ..constants import COMMAND_FILE_REGEX
from ..settings import get_root_dir, get_settings

CommandInstructionsType = TypeVar("CommandInstructionsType", bound=AnyInstructionModel)


class CommandFileInfo(BaseModel, Generic[CommandInstructionsType]):
    batch_id: BatchId
    path: Path
    type: BatchOpType
    model: CommandModel[CommandInstructionsType]

    def move_to_data_dir(self) -> None:
        data_dir = (
            get_root_dir() / Directories.Data / f"batch-{self.batch_id}" / "commands"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                self.path = self.path.rename(
                    data_dir / f"{self.type}_{self.model.command_id}.json"
                )
                break
            except FileExistsError:
                time.sleep(0.5)
        self.save()

    def reject(self, reason: CommandEndReason) -> None:
        self.model.start_time = datetime.now(timezone.utc)
        self.model.end_time = self.model.start_time
        self.model.end_reason = reason
        self.move_to_data_dir()

    def save(self) -> None:
        self.path.write_text(self.model.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def parse(file: Path) -> CommandFileInfo[AnyInstructionModel]:
        """
        :param file: The input file to parse
        :return: A CommandFileInfo data structure with the parsed information
        :raises FileNotFoundError: if the file does not exist
        :raises ValueError: if the file type is not named correctly
        """

        if not file.exists() or not file.is_file():
            raise FileNotFoundError(f"Command file {file} does not exist")
        groups = re.match(COMMAND_FILE_REGEX, file.name)
        if not groups:
            raise ValueError(
                f"Command file {file} does not contain the correct name format"
            )
        for _ in range(3):
            try:
                loaded_model = CommandModel[AnyInstructionModel].model_validate_json(
                    file.read_text(), context={"settings": get_settings()}
                )
                cap = BatchOpType(groups[2])
                return CommandFileInfo(
                    batch_id=groups[1], path=file, type=cap, model=loaded_model
                )
            except ValidationError as e:
                print(f"Error parsing command file {file}: {e}")
                time.sleep(0.5)

        raise ValueError(f"Failed to parse command file {file} after 3 attempts")


class CommandModel(BaseModel, Generic[CommandInstructionsType]):
    type: BatchOpType
    timestamp: datetime
    batch_id: BatchId
    command_id: str
    runs: int = Field(default=1)
    instructions: list[CommandInstructionsType]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    end_reason: Optional[CommandEndReason] = None

    def total_commands(self) -> int:
        return len(self.instructions) * self.runs

    def is_complete(self) -> bool:
        return self.end_time is not None and self.end_reason is not None

    @field_validator("instructions")
    def validate_instructions(
        cls, val: list[CommandInstructionsType], info: ValidationInfo
    ) -> list[CommandInstructionsType]:
        validator_name = f'[batch-{info.data["batch_id"]}-validator]: '
        if len(val) == 0:
            raise ValueError("Empty instruction list")
        list_type = type(val[0])
        for i in range(len(val)):
            list_item = val[i]
            if not isinstance(list_item, list_type):

                raise ValueError(
                    f"[{validator_name}] All items in a command list must be the same type. Expected: {list_type}, Actual: {type(list_item)}"
                )
            if not isinstance(list_item, TimedInstructions):
                continue
            if list_item.duration_ms is None and i != len(val) - 1:
                raise ValueError(
                    f"[{validator_name}] All timed instructions must have duration_ms set, excepting the last value (idx {i})"
                )

        return val
