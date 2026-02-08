from gfmtuner.algorithm.mcts.mcts_node import MCTSNode, MCTSNodeType
from gfmtuner.llm_call.openai_llm import call_openai
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import tempfile
import re
import json
import numpy as np


class GradientStore:

    def __init__(self):
        self.gradients: List[Dict] = []
        self.task_ids: List[str] = []

    def add_gradient(self, gradient: Dict[str, List], task_id: str = "unknown"):
        self.gradients.append(gradient)
        self.task_ids.append(task_id)
        print(f"[GradientStore] Stored gradient #{len(self.gradients)} for task {task_id}")
    
    def compute_similarity(self, new_gradient: Dict[str, List]) -> float:
        if not self.gradients:
            return -1.0
        
        similarities = []
        for stored_grad in self.gradients:
            sim = self._cosine_similarity(new_gradient, stored_grad)
            if sim is not None:
                similarities.append(sim)
        
        if not similarities:
            return -1.0
        
        return float(np.mean(similarities))
    
    def _cosine_similarity(self, grad1: Dict, grad2: Dict) -> Optional[float]:
        try:
            common_keys = set(grad1.keys()) & set(grad2.keys())
            if not common_keys:
                return None
            
            vec1 = []
            vec2 = []
            for key in sorted(common_keys):
                flat1 = self._flatten(grad1[key])
                flat2 = self._flatten(grad2[key])
                if len(flat1) == len(flat2):
                    vec1.extend(flat1)
                    vec2.extend(flat2)
            
            if not vec1:
                return None
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0
            
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            print(f"[GradientStore] Similarity computation error: {e}")
            return None
    
    def _flatten(self, nested: Any) -> List[float]:
        if isinstance(nested, (int, float)):
            return [float(nested)]
        elif isinstance(nested, list):
            result = []
            for item in nested:
                result.extend(self._flatten(item))
            return result
        return []
    
    def get_count(self) -> int:
        return len(self.gradients)


_gradient_store: Optional[GradientStore] = None


def get_gradient_store() -> GradientStore:
    global _gradient_store
    if _gradient_store is None:
        _gradient_store = GradientStore()
    return _gradient_store


def reset_gradient_store() -> None:
    global _gradient_store
    _gradient_store = GradientStore()
    print("[GradientStore] Reset for new task")


class RewardModel:
    def __init__(self, **kwargs):
        pass
    
    def get_reward(self, end_node: MCTSNode) -> float:
        raise NotImplementedError()


class LLMScoreRewardModel(RewardModel):

    def __init__(
        self,
        conda_env: str = "GFMTuner",
        graph_data_path: str = "your_graph_data_path_here",
        pretrained_model_path: str = "your_pretrained_model_path_here",
        execution_timeout: int = 120,
        max_score: float = 10.0,
        gradient_sample_k: int = 3,
    ):
        super().__init__()
        self.conda_env = conda_env
        self.graph_data_path = str(Path(graph_data_path).resolve())
        self.pretrained_model_path = str(Path(pretrained_model_path).resolve())
        self.execution_timeout = execution_timeout
        self.max_score = max_score
        self.gradient_sample_k = gradient_sample_k
    
    def get_reward(self, end_node: MCTSNode) -> float:
        prompt = self._build_prompt(end_node)
        response = call_openai(prompt, model="gpt-4")
        llm_score = self._extract_score(response)
        
        code = end_node.final_code or ""
        execution_result = self._execute_code_with_gradients(code)
        
        intermediate_results = self._get_intermediate_execution_results(end_node)
        
        gradient_consistency = -1.0
        if execution_result.get("success") and execution_result.get("gradients"):
            gradient_consistency = self._check_gradient_consistency(end_node)
        
        final_score = self._calculate_final_score(
            llm_score,
            execution_result,
            intermediate_results,
            gradient_consistency
        )
        
        print(f"[Reward] LLM={llm_score:.2f}, Exec={execution_result.get('success')}, "
              f"GradCons={gradient_consistency:.3f}, Final={final_score:.2f}")
        
        return final_score
    
    def _check_gradient_consistency(self, end_node: MCTSNode) -> float:
        if self.gradient_sample_k <= 1:
            return -1.0
        
        prompt = self._build_code_generation_prompt(end_node)
        
        gradients_list = []
        for i in range(self.gradient_sample_k):
            response = call_openai(prompt, model="gpt-4", temperature=0.7)
            code = self._extract_code_from_response(response)
            result = self._execute_code_with_gradients(code)
            
            if result.get("success") and result.get("gradients"):
                gradients_list.append(result["gradients"])
        
        if len(gradients_list) < 2:
            return -1.0
        
        similarities = []
        for i in range(len(gradients_list)):
            for j in range(i + 1, len(gradients_list)):
                grad_store = GradientStore()
                grad_store.add_gradient(gradients_list[i])
                sim = grad_store.compute_similarity(gradients_list[j])
                if sim >= 0:
                    similarities.append(sim)
        
        if not similarities:
            return -1.0
        
        return float(np.mean(similarities))
    
    def _build_code_generation_prompt(self, end_node: MCTSNode) -> str:
        return f"Generate GNN code based on: {end_node.original_query}"
    
    def _extract_code_from_response(self, response: str) -> str:
        match = re.search(r"```python\n(.*?)```", response, flags=re.DOTALL)
        if match:
            return match.group(1)
        return response
    
    def _execute_code_with_gradients(self, code: str) -> Dict[str, Any]:
        if not code:
            return {"success": False, "error": "No code", "gradients": None}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            try:
                self._copy_data_files_to_sandbox(tmpdir_path)
            except Exception as e:
                return {"success": False, "error": str(e), "gradients": None}
            
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
                
                gradients = None
                gradients_file = tmpdir_path / "gradients.json"
                if result.returncode == 0 and gradients_file.exists():
                    try:
                        with open(gradients_file, "r") as f:
                            gradients = json.load(f)
                    except Exception:
                        pass
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout[:1000] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "gradients": gradients,
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Timeout", "gradients": None}
            except Exception as e:
                return {"success": False, "error": str(e), "gradients": None}

    def _get_intermediate_execution_results(self, end_node: MCTSNode) -> Dict[str, Any]:
        results = {}
        
        for node in end_node.path_nodes:
            if node.node_type == MCTSNodeType.GRAPH_DATA_ANALYSIS:
                if hasattr(node, 'execution_result'):
                    results['data_analysis_success'] = node.execution_result.get('success', False)
        
        return results
    
    def _copy_data_files_to_sandbox(self, sandbox_dir: Path) -> None:
        import shutil
        
        src_graph = Path(self.graph_data_path)
        if src_graph.exists():
            dst_graph = sandbox_dir / "graph_data.pt"
            shutil.copy2(src_graph, dst_graph)
        
        src_model = Path(self.pretrained_model_path)
        if src_model.exists():
            dst_model = sandbox_dir / "graphmae_pretrained.pt"
            shutil.copy2(src_model, dst_model)

    def _calculate_final_score(
        self,
        llm_score: float,
        execution_result: Dict[str, Any],
        intermediate_results: Optional[Dict[str, Any]] = None,
        gradient_consistency: float = -1.0
    ) -> float:
        score = llm_score
        
        if intermediate_results:
            data_analysis_success = intermediate_results.get("data_analysis_success")
            if data_analysis_success is True:
                score = min(self.max_score, score + 0.5)
                print(f"[Reward] DataAnalysis execution succeeded, bonus +0.5")
            elif data_analysis_success is False:
                score *= 0.8
                print(f"[Reward] DataAnalysis execution failed, penalty *0.8")
        
        if execution_result.get("success"):
            pass
        else:
            score *= 0.3
        
        if gradient_consistency >= 0:
            if gradient_consistency >= 0.9:
                bonus = 1.5
                score = min(self.max_score, score + bonus)
                print(f"[Reward] Very high gradient consistency ({gradient_consistency:.3f}), bonus +{bonus}")
            elif gradient_consistency >= 0.7:
                bonus = 1.0
                score = min(self.max_score, score + bonus)
                print(f"[Reward] High gradient consistency ({gradient_consistency:.3f}), bonus +{bonus}")
            elif gradient_consistency >= 0.5:
                bonus = 0.5
                score = min(self.max_score, score + bonus)
                print(f"[Reward] Moderate gradient consistency ({gradient_consistency:.3f}), bonus +{bonus}")
            else:
                score *= 0.7
                print(f"[Reward] Low gradient consistency ({gradient_consistency:.3f}), penalty *0.7 (unstable path)")
        
        return score

    def _build_prompt(self, end_node: MCTSNode) -> str:
        path_summaries: List[str] = []
        for node in end_node.path_nodes:
            desc = f"{node.node_type.value}: "
            if node.node_type == MCTSNodeType.REQUIREMENT_DEFINITION:
                desc += f"requirement={node.requirement_summary}"
            elif node.node_type == MCTSNodeType.GRAPH_DATA_ANALYSIS:
                desc += f"data_analysis={node.data_analysis}"
            elif node.node_type == MCTSNodeType.ARCHITECTURE_REASONING:
                desc += f"arch_choice={node.architecture_reasoning}"
            elif node.node_type == MCTSNodeType.ARCHITECTURE_DESIGN:
                desc += f"arch_design={node.architecture_design}"
            elif node.node_type == MCTSNodeType.FEATURE_ENGINEERING:
                desc += f"feature_strategy={node.feature_strategy}"
            elif node.node_type == MCTSNodeType.TRAINING_STRATEGY:
                desc += f"training={node.training_strategy}"
            elif node.node_type == MCTSNodeType.EXPERIMENT_DESIGN:
                desc += f"experiment={node.experiment_design}"
            elif node.node_type == MCTSNodeType.CODE_IMPLEMENTATION:
                desc += "code_generated"
            elif node.node_type == MCTSNodeType.END:
                desc += "end"
            path_summaries.append(desc)

        code = end_node.final_code or ""
        query = end_node.original_query
        hint = end_node.hint
        context = end_node.context

        prompt = (
            "You are a strict GNN code judge. Score the solution from 0 to 10.\n"
            "Consider: task alignment, model/dataset/loss/metric suitability, code correctness, trainability, and likelihood of good performance.\n"
            "Return ONLY a JSON block like: ```json\n{\"score\": 7.5, \"brief_reason\": \"...\"}\n```.\n\n"
            f"User query: {query}\nHint: {hint}\nContext: {context}\n"
            f"Reasoning path: {' | '.join(path_summaries)}\n"
            f"Code:\n```python\n{code}\n```\n"
        )
        return prompt

    def _extract_score(self, text: str) -> float:
        if not text:
            return 0.0

        try:
            match = re.search(r"```json\n(.*?)```", text, flags=re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict) and "score" in parsed:
                    return float(parsed["score"])
        except Exception:
            pass

        match = re.search(r"([0-9]+\.?[0-9]*)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        return 0.0
