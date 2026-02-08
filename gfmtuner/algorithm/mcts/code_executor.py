
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import tempfile
import shutil


class CodeExecutor:

    def __init__(
        self,
        conda_env: str = "GFMTuner",
        graph_data_path: str = "graph_dataset/graph_data.pt",
        pretrained_model_path: Optional[str] = None,
        execution_timeout: int = 120,
    ):
        self.conda_env = conda_env
        self.graph_data_path = str(Path(graph_data_path).resolve())
        self.pretrained_model_path = str(Path(pretrained_model_path).resolve()) if pretrained_model_path else None
        self.execution_timeout = execution_timeout

    def execute(self, code: str, require_pretrained_model: bool = False) -> Dict[str, Any]:
        if not code or not code.strip():
            return {
                "success": False,
                "error": "No code provided",
                "stdout": "",
                "stderr": "Empty code string",
                "return_code": -1,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            

            try:
                self._copy_files_to_sandbox(tmpdir_path, require_pretrained_model)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to setup sandbox: {str(e)[:300]}",
                    "stdout": "",
                    "stderr": str(e),
                    "return_code": -1,
                }
            

            code_file = tmpdir_path / "run_code.py"
            code_file.write_text(code)


            cmd = [
                "conda", "run", "-n", self.conda_env,
                "--no-capture-output",
                "python", str(code_file)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout,
                    cwd=tmpdir,
                )
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout[:5000] if result.stdout else "",
                    "stderr": result.stderr[:2000] if result.stderr else "",
                    "return_code": result.returncode,
                    "error": None if result.returncode == 0 else result.stderr[:500],
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Execution timeout ({self.execution_timeout}s)",
                    "stdout": "",
                    "stderr": "Timeout",
                    "return_code": -1,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)[:500],
                    "stdout": "",
                    "stderr": str(e),
                    "return_code": -1,
                }

    def _copy_files_to_sandbox(self, sandbox_dir: Path, require_pretrained_model: bool) -> None:

        src_graph = Path(self.graph_data_path)
        if src_graph.exists():
            dst_graph = sandbox_dir / "graph_data.pt"
            shutil.copy2(src_graph, dst_graph)
            print(f"[Sandbox] Copied {src_graph} -> {dst_graph}")
        else:
            raise FileNotFoundError(f"Graph data file not found: {src_graph}")
        

        if require_pretrained_model and self.pretrained_model_path:
            src_model = Path(self.pretrained_model_path)
            if src_model.exists():
                dst_model = sandbox_dir / "graphmae_pretrained.pt"
                shutil.copy2(src_model, dst_model)
                print(f"[Sandbox] Copied {src_model} -> {dst_model}")
            else:
                raise FileNotFoundError(f"Pretrained model file not found: {src_model}")



_global_executor: Optional[CodeExecutor] = None


def get_code_executor() -> Optional[CodeExecutor]:
    return _global_executor


def set_code_executor(executor: CodeExecutor) -> None:
    global _global_executor
    _global_executor = executor
