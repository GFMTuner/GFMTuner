from gfmtuner.llm_call.prompt_factory import get_prompt
from gfmtuner.llm_call.openai_llm import call_openai
from typing import Dict, Any, List, Optional
from enum import Enum
import copy
import json
import re


def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"```json\n(.*?)```", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def _extract_python_code(text: str) -> Optional[str]:
    try:
        return re.search(r"```python\n(.*?)```", text, flags=re.DOTALL).group(1)
    except Exception:
        return text.strip() if text else None


class MCTSAction:
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        raise NotImplementedError()


class RequirementDefinitionAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="requirement_definition_gnn",
            template_args={
                "QUERY": node.original_query,
                "HINT": node.hint or "",
                "CONTEXT": node.context,
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}

            if not parsed and response:
                parsed = {"raw": response}
            elif response and "code" not in parsed:
                parsed.setdefault("raw", response)
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.REQUIREMENT_DEFINITION
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.requirement_summary = parsed
            nodes.append(child_node)
        return nodes


class GraphDataAnalysisAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        from gfmtuner.algorithm.mcts.code_executor import get_code_executor
        
        prompt = get_prompt(
            template_name="graph_data_analysis_gnn",
            template_args={
                "QUERY": node.original_query,
                "HINT": node.hint or "",
                "REQUIREMENT": json.dumps(node.requirement_summary or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:

            code = _extract_python_code(response)
            

            execution_result = None
            executor = get_code_executor()
            if executor and code:
                print(f"[GraphDataAnalysis] Executing data analysis code in sandbox...")
                execution_result = executor.execute(code, require_pretrained_model=False)
                if execution_result.get("success"):
                    print(f"[GraphDataAnalysis] ✅ Code executed successfully")
                    print(f"[GraphDataAnalysis] Output:\n{execution_result.get('stdout', '')[:1000]}")
                else:
                    print(f"[GraphDataAnalysis] ❌ Code execution failed: {execution_result.get('error', '')[:200]}")
            

            parsed = _parse_json_block(response) or {}
            if not parsed and response:
                parsed = {"raw": response}
            

            if code:
                parsed["code"] = code
            if execution_result:
                parsed["execution_result"] = execution_result

                if execution_result.get("success") and execution_result.get("stdout"):
                    parsed["data_insights"] = execution_result["stdout"]
            
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.GRAPH_DATA_ANALYSIS
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.data_analysis = parsed
            nodes.append(child_node)
        return nodes


class ArchitectureReasoningAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="architecture_reasoning_gnn",
            template_args={
                "QUERY": node.original_query,
                "REQUIREMENT": json.dumps(node.requirement_summary or {}, ensure_ascii=False),
                "DATA_ANALYSIS": json.dumps(node.data_analysis or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.ARCHITECTURE_REASONING
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.architecture_reasoning = parsed
            nodes.append(child_node)
        return nodes


class ArchitectureDesignAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="architecture_design_gnn",
            template_args={
                "QUERY": node.original_query,
                "ARCH_REASONING": json.dumps(node.architecture_reasoning or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.ARCHITECTURE_DESIGN
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.architecture_design = parsed
            nodes.append(child_node)
        return nodes


class FeatureEngineeringAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="feature_engineering_gnn",
            template_args={
                "QUERY": node.original_query,
                "DATA_ANALYSIS": json.dumps(node.data_analysis or {}, ensure_ascii=False),
                "ARCHITECTURE_DESIGN": json.dumps(node.architecture_design or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.FEATURE_ENGINEERING
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.feature_strategy = parsed
            nodes.append(child_node)
        return nodes


class TrainingStrategyAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="training_strategy_gnn",
            template_args={
                "QUERY": node.original_query,
                "DATA_ANALYSIS": json.dumps(node.data_analysis or {}, ensure_ascii=False),
                "FEATURE_STRATEGY": json.dumps(node.feature_strategy or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.TRAINING_STRATEGY
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.training_strategy = parsed
            nodes.append(child_node)
        return nodes


class ExperimentDesignAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        prompt = get_prompt(
            template_name="experiment_design_gnn",
            template_args={
                "QUERY": node.original_query,
                "TRAINING_STRATEGY": json.dumps(node.training_strategy or {}, ensure_ascii=False),
            },
        )
        responses = call_openai(prompt, **llm_kwargs)
        nodes: List["MCTSNode"] = []
        for response in responses:
            parsed = _parse_json_block(response) or {}
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.EXPERIMENT_DESIGN
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.experiment_design = parsed
            nodes.append(child_node)
        return nodes


class CodeImplementationAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:

        data_insights = ""
        if node.data_analysis:

            if "data_insights" in node.data_analysis:
                data_insights = node.data_analysis["data_insights"]
            elif "execution_result" in node.data_analysis:
                exec_result = node.data_analysis["execution_result"]
                if exec_result.get("success") and exec_result.get("stdout"):
                    data_insights = exec_result["stdout"]
        
        if data_insights:
            print(f"[CodeImplementation] Using real graph data insights ({len(data_insights)} chars)")
        else:
            print(f"[CodeImplementation] No data insights available, using defaults")
        
        prompt = get_prompt(
            template_name="code_implementation_gnn",
            template_args={
                "QUERY": node.original_query,
                "HINT": node.hint or "",
                "DATA_INSIGHTS": data_insights,
                "ARCHITECTURE_DESIGN": json.dumps(node.architecture_design or {}, ensure_ascii=False),
                "TRAINING_STRATEGY": json.dumps(node.training_strategy or {}, ensure_ascii=False),
                "FEATURE_STRATEGY": json.dumps(node.feature_strategy or {}, ensure_ascii=False),
                "EXPERIMENT_DESIGN": json.dumps(node.experiment_design or {}, ensure_ascii=False),
            },
        )

        child_node = copy.deepcopy(node)
        child_node.node_type = MCTSNodeType.CODE_IMPLEMENTATION
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1
        child_node.children = []
        child_node.path_nodes = node.path_nodes + [child_node]

        code = None
        parsed_plan: Dict[str, Any] = {}
        while code is None:
            responses = call_openai(prompt, **llm_kwargs)
            candidate = responses[0] if responses else ""
            parsed_plan = _parse_json_block(candidate) or {}
            maybe_code = parsed_plan.get("code") or candidate
            code = _extract_python_code(maybe_code)
        child_node.code_plan = parsed_plan
        child_node.generated_code = code
        return [child_node]


class EndAction(MCTSAction):
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        child_node = copy.deepcopy(node)
        child_node.node_type = MCTSNodeType.END
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1
        child_node.children = []
        child_node.path_nodes = node.path_nodes + [child_node]
        child_node.final_code = node.reviewed_code or node.generated_code
        return [child_node]


class MCTSNodeType(Enum):
    ROOT = "root"
    REQUIREMENT_DEFINITION = "requirement_definition"
    GRAPH_DATA_ANALYSIS = "graph_data_analysis"
    ARCHITECTURE_REASONING = "architecture_reasoning"
    ARCHITECTURE_DESIGN = "architecture_design"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING_STRATEGY = "training_strategy"
    EXPERIMENT_DESIGN = "experiment_design"
    CODE_IMPLEMENTATION = "code_implementation"
    END = "end"


NODE_TYPE_TO_VALID_ACTIONS = {
    MCTSNodeType.ROOT: [RequirementDefinitionAction],
    MCTSNodeType.REQUIREMENT_DEFINITION: [GraphDataAnalysisAction],
    MCTSNodeType.GRAPH_DATA_ANALYSIS: [ArchitectureReasoningAction],
    MCTSNodeType.ARCHITECTURE_REASONING: [ArchitectureDesignAction],
    MCTSNodeType.ARCHITECTURE_DESIGN: [FeatureEngineeringAction],
    MCTSNodeType.FEATURE_ENGINEERING: [TrainingStrategyAction],
    MCTSNodeType.TRAINING_STRATEGY: [ExperimentDesignAction],
    MCTSNodeType.EXPERIMENT_DESIGN: [CodeImplementationAction],
    MCTSNodeType.CODE_IMPLEMENTATION: [EndAction],
    MCTSNodeType.END: [],
}
