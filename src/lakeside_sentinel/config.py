from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Google Nest authentication
    google_master_token: str
    google_username: str
    nest_device_id: str

    # Resend email
    resend_api_key: str
    alert_email_to: str
    alert_email_from: str = "alerts@xeroshot.org"

    # Object Detection Model
    yolo_model: str = "yolo26s.pt"
    yolo_batch_size: int = 16
    crop_padding: float = 0.2

    # Region of Interest (fraction 0.0–1.0)
    roi_y_start: float = 0.0
    roi_y_end: float = 1.0
    roi_x_start: float = 0.0
    roi_x_end: float = 1.0

    # Camera location (for daylight filtering)
    camera_latitude: float
    camera_longitude: float

    # Vehicle Detection (VEH)
    veh_confidence_threshold: float = 0.4
    veh_fps_sample: int = 2

    # High-speed person detection (HSP)
    hsp_fps_sample: int = 4
    hsp_displacement_threshold: float = 60.0
    hsp_person_confidence_threshold: float = 0.4
    hsp_max_match_distance: float = 200.0
