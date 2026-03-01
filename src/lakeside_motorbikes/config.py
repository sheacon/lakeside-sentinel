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

    # Detection
    yolo_confidence_threshold: float = 0.4
    crop_padding: float = 0.2

    # Frame sampling
    fps_sample: int = 2

    # Polling
    poll_interval_seconds: int = 120
