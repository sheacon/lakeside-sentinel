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
    yolo_model: str = "yolo_models/yolo26s.pt"
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
    hsp_displacement_threshold: float = 240.0
    hsp_person_confidence_threshold: float = 0.4
    hsp_max_match_distance: float = 800.0

    # Review server
    review_port: int = 5000

    # Claude Vision verification (optional)
    anthropic_api_key: str = ""
    claude_vision_model: str = "claude-sonnet-4-20250514"
    claude_vision_prompt: str = (
        "Is there a motorized two-wheeled vehicle in this image — such as a motorcycle, "
        "motorbike, scooter, moped, or e-bike? People may be riding or standing near it. "
        "Do NOT count baby strollers, prams, pushchairs, wagons, wheelchairs, "
        "shopping carts, or any non-motorized wheeled object. "
        'Answer "yes" or "no" only. '
        'Answer "yes" only if a motorized two-wheeled vehicle is clearly visible. '
        'Answer "no" for everything else.'
    )
