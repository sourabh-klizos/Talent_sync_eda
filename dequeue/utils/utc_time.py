from datetime import datetime

def get_current_time_utc() -> str:
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')