from gfmtuner.algorithm.mcts.mcts_node import *
from gfmtuner.algorithm.mcts.mcts_action import *
from gfmtuner.algorithm.mcts.reward import *
from gfmtuner.runner.task import Task
import math
import random
from pathlib import Path
from typing import Dict, Any, List
import pickle

class MCTSSolver:
    def __init__(self,
                 task: Task, 
                 max_rollout_steps: int,
                 exploration_constant: float,
                 save_root_dir: str,
                 llm_kwargs: Dict[str, Any],
                 reward_model: RewardModel):
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.task = task
        self.max_rollout_steps = max_rollout_steps
        self.exploration_constant = exploration_constant
        self.save_root_dir = save_root_dir
    
    def select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.is_terminal():
            if not all(child.N > 0 for child in current.children):
                return next(child for child in current.children if child.N == 0)
            
            current = max(current.children, key=lambda child: (child.Q / child.N) + self.exploration_constant * math.sqrt(math.log(current.N) / child.N))
        return current
    
    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        assert node.children == [], f"Children nodes of node {node.node_type} before expansion is not empty"
        valid_action_space = get_valid_action_space_for_node(node)
        for action in valid_action_space:
            action_nodes = action.create_children_nodes(node, self.llm_kwargs)

            for child in action_nodes:
                print(
                    f"[Expand] task={node.task_id} depth={child.depth} parent={node.node_type.value} -> {child.node_type.value} via {action.__class__.__name__}"
                )
            node.children.extend(action_nodes)
        
        random.shuffle(node.children)
        


    def simulate(self, node: MCTSNode) -> MCTSNode:
        assert node.children == [], f"Node before simulation have non-empty children"
        current = node
        while not current.is_terminal():
            self.expand(current)
            current = random.choice(current.children)
        return current

    def backpropagate(self, node: MCTSNode):
        print("Backpropagate, Final Code Metric: ", node.metric_score)
        current = node
        if current.N == 0:
            reward = self.reward_model.get_reward(current)
        else:
            reward = current.Q / current.N
        while current is not None:
            current.N += 1
            current.Q += reward
            current = current.parent_node
    
    def find_all_end_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        if node.node_type == MCTSNodeType.END:
            return [node]
        else:
            end_nodes = []
            for child in node.children:
                end_nodes.extend(self.find_all_end_nodes(child))
            return end_nodes
    
    def find_all_valid_reasoning_paths(self, node: MCTSNode) -> List[List[MCTSNode]]:
        end_nodes = self.find_all_end_nodes(node)
        reasoning_paths = []
        for end_node in end_nodes:
            reasoning_paths.append(end_node.path_nodes)
        return reasoning_paths
    
    def solve(self):
        root_node = MCTSNode(MCTSNodeType.ROOT,
                             parent_node=None,
                             parent_action=None,
                             depth=0,
                             task_id=self.task.task_id,
                             original_query=self.task.query,
                             hint=self.task.hint or "",
                             context=self.task.context or "")
        root_node.path_nodes = [root_node]
        
        for _ in range(self.max_rollout_steps):
            print(f"Task ID: {self.task.task_id}, Rollout step {_ + 1} / {self.max_rollout_steps}")
            leaf_node = self.select(root_node)
            if leaf_node.is_terminal():
                self.backpropagate(leaf_node)
                continue
            self.expand(leaf_node)
            leaf_node = random.choice(leaf_node.children)
            end_node = self.simulate(leaf_node)
            self.backpropagate(end_node)
        
        all_valid_reasoning_paths = self.find_all_valid_reasoning_paths(root_node)
        save_path = Path(self.save_root_dir) / f"{self.task.task_id}.pkl"
        print(f"Task ID: {self.task.task_id} done, Number of valid reasoning paths: {len(all_valid_reasoning_paths)}")
        with open(save_path, "wb") as f:
            pickle.dump(all_valid_reasoning_paths, f)

        