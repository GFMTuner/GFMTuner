from gfmtuner.algorithm.mcts.mcts import MCTSSolver
from gfmtuner.algorithm.mcts.reward import LLMScoreRewardModel
from gfmtuner.runner.task import Task
from gfmtuner.config.mcts_config import MCTSConfig
from pathlib import Path
from typing import Union
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
from gfmtuner.llm_call.openai_llm import DEFAULT_COST_RECORDER
import json
import random
from dotenv import load_dotenv  
import os
import traceback

load_dotenv(override=True)

try:
    import weave
    if "WANDB_API_KEY" in os.environ:
        weave.init("mcts_runner")
    else:
        print("WANDB_API_KEY is not set in environment variables, will not use it")
except ImportError:
    print("Weave is not installed, will not use it")

class MCTSRunner:
    def __init__(self, config: Union[MCTSConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = MCTSConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = MCTSConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config
            
        if not Path(self.config.save_root_dir).exists():
            Path(self.config.save_root_dir).mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)
        

        self._init_code_executor()
        
    def _init_code_executor(self):
        from gfmtuner.algorithm.mcts.code_executor import CodeExecutor, set_code_executor
        
        exec_config = self.config.code_execution
        if exec_config and exec_config.enable_execution:
            executor = CodeExecutor(
                conda_env=exec_config.conda_env,
                graph_data_path=exec_config.graph_data_path,
                pretrained_model_path=exec_config.pretrained_model_path,
                execution_timeout=exec_config.execution_timeout,
            )
            set_code_executor(executor)
            print(f"[MCTSRunner] Code executor initialized (conda_env={exec_config.conda_env})")
        else:
            set_code_executor(None)
            print("[MCTSRunner] Code execution disabled")
        
    def run_one_task(self, task: Task) -> str:
        reward_kwargs = self.config.reward_model_kwargs or self.config.mcts_model_kwargs
        

        exec_config = self.config.code_execution
        if exec_config:
            reward_model = LLMScoreRewardModel(
                llm_kwargs=reward_kwargs,
                conda_env=exec_config.conda_env,
                graph_data_path=exec_config.graph_data_path,
                pretrained_model_path=exec_config.pretrained_model_path,
                execution_timeout=exec_config.execution_timeout,
                enable_execution=exec_config.enable_execution,
                gradient_sample_count=exec_config.gradient_sample_count,
            )
        else:
            reward_model = LLMScoreRewardModel(
                llm_kwargs=reward_kwargs,
                enable_execution=False,
            )
        
        mcts_solver = MCTSSolver(
            task=task,
            max_rollout_steps=self.config.max_rollout_steps,
            exploration_constant=self.config.exploration_constant,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            reward_model=reward_model
        )
        try:
            mcts_solver.solve()
        except Exception as e:
            print("-" * 100)
            print(f"Error solving task {task.task_id}: {e}")
            traceback.print_exc()
            print(f"The task {task.task_id} has been given up")
            print("-" * 100)
        DEFAULT_COST_RECORDER.print_profile()
    
    def run_all_tasks(self):
        with open(self.config.tasks_file_path, "rb") as f:
            tasks = pickle.load(f)
            
        if self.config.subset_file_path:
            print(f"Using subset file {self.config.subset_file_path} to filter tasks")
            with open(self.config.subset_file_path, "r") as f:
                subset_data = json.load(f)
                subset_ids = [item.get("task_id") or item.get("question_id") for item in subset_data]
                tasks = [task for task in tasks if task.task_id in subset_ids]
            print(f"Filtered {len(tasks)} tasks from {len(tasks)} tasks")
        
        done_task_ids = []
        for pkl_file in Path(self.config.save_root_dir).glob("*.pkl"):
            done_task_ids.append(pkl_file.stem)
        print(f"Ignore done task ids: {done_task_ids}")
        tasks = [task for task in tasks if str(task.task_id) not in done_task_ids]
        
        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            print(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(self.config.model_dump(), f, indent=4)

        print(f"There are {len(tasks)} tasks to solve")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m gfmtuner.runner.mcts_runner <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    runner = MCTSRunner(config=config_path)
    runner.run_all_tasks()
