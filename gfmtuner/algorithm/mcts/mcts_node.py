from typing import List, Optional, Dict, Any
from gfmtuner.algorithm.mcts.mcts_action import *


def get_valid_action_space_for_node(node: "MCTSNode") -> List["MCTSAction"]:

    prefix_actions = [
        RequirementDefinitionAction,
        GraphDataAnalysisAction,
        ArchitectureReasoningAction,
        ArchitectureDesignAction,
        FeatureEngineeringAction,
        TrainingStrategyAction,
        ExperimentDesignAction,
    ]
    used = {path_node.parent_action.__class__ for path_node in node.path_nodes if path_node.parent_action is not None}
    impl_done = CodeImplementationAction in used


    if impl_done:
        return [EndAction()]


    remaining_prefix = [cls for cls in prefix_actions if cls not in used]
    if remaining_prefix:
        return [cls() for cls in remaining_prefix]
    return [CodeImplementationAction()]


class MCTSNode:
    def __init__(self,
                 node_type: "MCTSNodeType",
                 parent_node: Optional["MCTSNode"] = None,
                 parent_action: Optional["MCTSAction"] = None,
                 depth: int = 0,
                 task_id: str = "",
                 original_query: str = "",
                 hint: str = "",
                 context: str = "",
                 path_nodes: Optional[List["MCTSNode"]] = None,
                 requirement_summary: Optional[Dict[str, Any]] = None,
                 data_analysis: Optional[Dict[str, Any]] = None,
                 architecture_reasoning: Optional[Dict[str, Any]] = None,
                 architecture_design: Optional[Dict[str, Any]] = None,
                 feature_strategy: Optional[Dict[str, Any]] = None,
                 training_strategy: Optional[Dict[str, Any]] = None,
                 experiment_design: Optional[Dict[str, Any]] = None,
                 code_plan: Optional[Dict[str, Any]] = None,
                 generated_code: Optional[str] = None,
                 reviewed_code: Optional[str] = None,
                 final_code: Optional[str] = None,
                 metric_score: Optional[float] = None,
                 evaluation_log: Optional[str] = None,
                 execution_result: Optional[Dict[str, Any]] = None,
                 llm_kwargs: Optional[Dict[str, Any]] = None):
        self.node_type = node_type
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.depth = depth
        self.task_id = task_id
        self.original_query = original_query
        self.hint = hint
        self.context = context or ""
        self.children: List[MCTSNode] = []
        self.path_nodes = path_nodes or []

        self.requirement_summary = requirement_summary
        self.data_analysis = data_analysis
        self.architecture_reasoning = architecture_reasoning
        self.architecture_design = architecture_design
        self.feature_strategy = feature_strategy
        self.training_strategy = training_strategy
        self.experiment_design = experiment_design
        self.code_plan = code_plan
        self.generated_code = generated_code
        self.reviewed_code = reviewed_code
        self.final_code = final_code
        self.metric_score = metric_score
        self.evaluation_log = evaluation_log
        self.execution_result = execution_result
        self.llm_kwargs = llm_kwargs

        self.Q = 0
        self.N = 0
    
    def create_children(self):
        if self.children:
            return
        valid_action_space = get_valid_action_space_for_node(self)
        for action in valid_action_space:
            self.children.extend(action.create_children_nodes(self, self.llm_kwargs))
            
    def is_terminal(self):
        return self.node_type == MCTSNodeType.END


