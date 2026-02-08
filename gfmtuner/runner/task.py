from pydantic import BaseModel
from typing import Optional, Dict, Any

class Task(BaseModel):
    task_id: str
    query: str
    hint: Optional[str] = None
    context: Optional[str] = None
    seed: Optional[int] = 42
    user_preferences: Optional[Dict[str, Any]] = None
    artifacts_root: Optional[str] = None