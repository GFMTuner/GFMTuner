from pathlib import Path
import sys
import pickle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfmtuner.runner.task import Task


def main():
    tasks = [
        Task(
            task_id="demo-1",
            query="your_query_here",
            hint="your_hint_here",
            context="your_context_here",
            user_preferences={"prefer_model": "GCN"},
        )
    ]
    out_path = Path("data/GFMTuner/tasks.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(tasks, f)
    print(f"Wrote {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()
