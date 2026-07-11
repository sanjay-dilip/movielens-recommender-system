import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.evaluation import EVAL_SEEDS, K, N_NEGATIVES, format_markdown_table, run_unified_evaluation


def main():
    print("Running unified ALS + Hybrid evaluation...")
    raw_df, summary_df = run_unified_evaluation(seeds=EVAL_SEEDS, k=K, n_negatives=N_NEGATIVES)

    print("\nPer-seed results:\n", raw_df)
    print("\nSummary (mean, std over seeds):\n", summary_df)

    table = format_markdown_table(summary_df, k=K)
    print("\nMarkdown table for README:\n")
    print(table)


if __name__ == "__main__":
    main()
