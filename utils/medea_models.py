from typing import List, Optional
from pydantic import BaseModel

from .models.batch_models import BatchId
from .models.command_models import CommandModel

class InterventionModel(BaseModel):
    batch_id: BatchId
    intervention_id: str
    result_video_path: Optional[str] = None
    pre_velocity: Optional[float] = None
    commands: List[CommandModel]