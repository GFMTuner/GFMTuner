from pydantic import BaseModel
from typing import Dict, Any, Optional

class CodeExecutionConfig(BaseModel):
    conda_env: str = "GFMTuner"
    graph_data_path: str = "your_graph_data_path_here"
    pretrained_model_path: str = "your_pretrained_model_path_here"
    execution_timeout: int = 120
    enable_execution: bool = True

    gradient_sample_count: int = 3

class MCTSConfig(BaseModel):
    """
    Configuration for the MCTS Runner.
    """
    tasks_file_path: str
    subset_file_path: Optional[str] = None
    resource_root_dir: Optional[str] = None
    n_processes: int = 1
    max_rollout_steps: int = 3
    exploration_constant: float = 1.414
    save_root_dir: str
    mcts_model_kwargs: Dict[str, Any]
    reward_model_kwargs: Optional[Dict[str, Any]] = None
    code_execution: Optional[CodeExecutionConfig] = None
    random_seed: Optional[int] = 42