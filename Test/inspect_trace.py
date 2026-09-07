"""
Fetches the most recent run from your LangSmith project and prints its
raw fields, so you can see for yourself there is no "reasoning" or
"because" field anywhere -- just inputs, outputs, and metadata.

Run agent.py first, wait a few seconds for LangSmith to ingest the
trace, then run this.
"""

import os
import json
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "trace-demo")


def main():
    client = Client()

    runs = list(
        client.list_runs(
            project_name=PROJECT_NAME,
            is_root=True,          # top-level run (the whole graph invocation)
            limit=1,
        )
    )

    if not runs:
        print(f"No runs found in project '{PROJECT_NAME}'. "
              "Did you run agent.py with LANGSMITH_TRACING_V2=true first?")
        return

    root_run = runs[0]
    print("=" * 70)
    print("ROOT RUN (the whole agent invocation)")
    print("=" * 70)
    print(f"name:       {root_run.name}")
    print(f"run_type:   {root_run.run_type}")
    print(f"inputs:     {json.dumps(root_run.inputs, indent=2, default=str)}")
    print(f"outputs:    {json.dumps(root_run.outputs, indent=2, default=str)}")
    print(f"start_time: {root_run.start_time}")
    print(f"end_time:   {root_run.end_time}")

    print("\n" + "=" * 70)
    print("CHILD RUNS (each individual step -- memory_lookup, tool, LLM call)")
    print("=" * 70)

    child_runs = list(client.list_runs(project_name=PROJECT_NAME, parent_run_id=root_run.id))
    # also fetch grandchildren (LangGraph nests a level deeper)
    for run in sorted(child_runs, key=lambda r: r.start_time):
        print(f"\n--- step: {run.name}  (run_type={run.run_type}) ---")
        print(f"inputs:  {json.dumps(run.inputs, indent=2, default=str)}")
        print(f"outputs: {json.dumps(run.outputs, indent=2, default=str)}")

    print("\n" + "=" * 70)
    print("Notice: every field above is a fact (input/output/timestamp).")
    print("Nowhere does the trace say WHY a step happened -- that 'why'")
    print("does not exist until your own Explainer generates it separately.")
    print("=" * 70)


if __name__ == "__main__":
    main()