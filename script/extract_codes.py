

import argparse
import pickle
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfmtuner.algorithm.mcts.mcts_node import MCTSNodeType


def extract_codes_from_pkl(pkl_path: Path, output_dir: Path = None):
    
    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)
    

    with open(pkl_path, 'rb') as f:
        paths = pickle.load(f)
    
    if not paths:
        print("No paths found in pkl file.")
        return
    

    if output_dir is None:
        output_dir = pkl_path.parent / "codes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(paths)} paths in {pkl_path}")
    print(f"Extracting codes to {output_dir}")
    print("-" * 60)
    
    for path_idx, path in enumerate(paths, 1):
        path_dir = output_dir / f"path_{path_idx}"
        path_dir.mkdir(exist_ok=True)
        
        codes_extracted = []
        execution_result = None
        
        for node in path:

            if node.node_type == MCTSNodeType.GRAPH_DATA_ANALYSIS:
                if node.data_analysis and isinstance(node.data_analysis, dict):

                    code = node.data_analysis.get("code", "") or node.data_analysis.get("raw", "")
                    if code:

                        code = _clean_code(code)
                        code_path = path_dir / "data_analysis.py"
                        code_path.write_text(code)
                        codes_extracted.append(("data_analysis.py", len(code)))
            

            if node.node_type == MCTSNodeType.CODE_IMPLEMENTATION:
                if node.generated_code:
                    code = _clean_code(node.generated_code)
                    code_path = path_dir / "generated_code.py"
                    code_path.write_text(code)
                    codes_extracted.append(("generated_code.py", len(code)))
            

            if node.node_type == MCTSNodeType.END:
                if node.final_code:
                    code = _clean_code(node.final_code)
                    code_path = path_dir / "final_code.py"
                    code_path.write_text(code)
                    codes_extracted.append(("final_code.py", len(code)))
                

                if node.metric_score is not None:
                    score_path = path_dir / "score.txt"
                    score_path.write_text(f"metric_score: {node.metric_score}\n")
                

                if hasattr(node, 'execution_result') and node.execution_result:
                    execution_result = node.execution_result
        

        if execution_result is not None:
            exec_file = path_dir / "execution_result.json"
            exec_file.write_text(json.dumps(execution_result, indent=2, ensure_ascii=False))
        

        if codes_extracted:
            print(f"Path {path_idx}:")
            for filename, size in codes_extracted:
                print(f"  - {filename} ({size} chars)")

            if execution_result is not None:
                if execution_result.get("skipped"):
                    print(f"  - Execution: ⏭️  SKIPPED")
                elif execution_result.get("success"):
                    print(f"  - Execution: ✅ SUCCESS")
                else:
                    print(f"  - Execution: ❌ FAILED")
                    error_msg = execution_result.get("error", "Unknown error")
                    if error_msg:
                        print(f"    Error: {error_msg[:100]}...")
        else:
            print(f"Path {path_idx}: No code extracted")
    
    print("-" * 60)
    print(f"Done! Codes saved to {output_dir}")


def _clean_code(code: str) -> str:
    if not code:
        return ""
    
    code = code.strip()
    

    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    

    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extract generated codes from MCTS result pkl file"
    )
    parser.add_argument(
        "pkl_path",
        type=str,
        help="Path to the MCTS result pkl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save extracted codes (default is a 'codes' folder in the same directory as the pkl file)"
    )
    
    args = parser.parse_args()
    
    pkl_path = Path(args.pkl_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    extract_codes_from_pkl(pkl_path, output_dir)


if __name__ == "__main__":
    main()
