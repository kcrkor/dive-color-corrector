"""Pydantic models for data validation."""

from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_video_path: str = Field(..., description="Path to input video file")
    output_video_path: str = Field(..., description="Path to output video file")
    fps: int = Field(..., gt=0, description="Frames per second")
    frame_count: int = Field(..., gt=0, description="Total frame count")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    filter_indices: list[int] = Field(
        default_factory=list, description="Frame indices for sampled filters"
    )
    filter_matrices: NDArray[Any] = Field(..., description="Filter matrices array")

    @field_validator("input_video_path")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Input video file does not exist: {v}")
        return v

    @field_validator("filter_matrices")
    @classmethod
    def validate_filter_matrices(cls, v: NDArray[Any]) -> NDArray[Any]:
        if v.ndim == 2 and v.shape[1] != 20 and v.size > 0:
            raise ValueError(f"Filter matrices must have shape (n, 20), got {v.shape}")
        return v
