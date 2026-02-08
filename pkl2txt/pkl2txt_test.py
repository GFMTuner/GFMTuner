import pickle, json, textwrap
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfmtuner.algorithm.mcts.mcts_node import MCTSNodeType

pkl_path = Path('results/GFMTuner/demo/demo-1.pkl')
paths = pickle.load(open(pkl_path, 'rb'))

out_lines = []
for idx, path in enumerate(paths, 1):
    out_lines.append(f"PATH {idx}\n" + "-"*40)
    for node in path:
        out_lines.append(f"  depth={node.depth} type={node.node_type.value}")
        if node.node_type == MCTSNodeType.REQUIREMENT_DEFINITION:
            out_lines.append(f"    requirement: {node.requirement_summary}")
        if node.node_type == MCTSNodeType.GRAPH_DATA_ANALYSIS:
            out_lines.append(f"    data_analysis: {node.data_analysis}")
        if node.node_type == MCTSNodeType.ARCHITECTURE_REASONING:
            out_lines.append(f"    arch_reasoning: {node.architecture_reasoning}")
        if node.node_type == MCTSNodeType.ARCHITECTURE_DESIGN:
            out_lines.append(f"    arch_design: {node.architecture_design}")
        if node.node_type == MCTSNodeType.FEATURE_ENGINEERING:
            out_lines.append(f"    feature_strategy: {node.feature_strategy}")
        if node.node_type == MCTSNodeType.TRAINING_STRATEGY:
            out_lines.append(f"    training_strategy: {node.training_strategy}")
        if node.node_type == MCTSNodeType.EXPERIMENT_DESIGN:
            out_lines.append(f"    experiment: {node.experiment_design}")
        if node.node_type == MCTSNodeType.CODE_IMPLEMENTATION:
            out_lines.append("    code (truncated 3000 chars):")
            code = (node.generated_code or '')[:3000].splitlines()
            out_lines.extend(["      " + line for line in code])
        if node.node_type == MCTSNodeType.CODE_REVIEW:
            out_lines.append("    reviewed code (truncated 3000 chars):")
            code = (node.reviewed_code or '')[:3000].splitlines()
            out_lines.extend(["      " + line for line in code])
        if node.node_type == MCTSNodeType.END:
            out_lines.append(f"    final code metric_score: {node.metric_score}")
    out_lines.append("")

out_path = Path('results/GFMTuner/demo/demo-1.txt')
out_path.write_text("\n".join(out_lines))
print(f"Wrote {len(paths)} paths to {out_path}")