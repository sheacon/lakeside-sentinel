from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CameraEvent:
    """A single motion event from the Nest camera."""

    start_time: datetime
    duration: timedelta

    @property
    def end_time(self) -> datetime:
        return self.start_time + self.duration

    @property
    def event_id(self) -> str:
        """Unique identifier based on start time (milliseconds since epoch)."""
        return str(int(self.start_time.timestamp() * 1000))
