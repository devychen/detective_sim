# =================================================
# descriptive_stats.py
# Simple descriptive overview of reference and simulation data
# =================================================

import glob
import os
import pandas as pd

# =================================================
# Paths
# =================================================

GOLD_PATH = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"


# =================================================
# Reference Corpus Stats
# =================================================

def reference_stats():

    print("\n==============================")
    print("REFERENCE CORPUS STATISTICS")
    print("==============================")

    if not os.path.exists(GOLD_PATH):
        print("Reference file not found.")
        return

    df = pd.read_csv(GOLD_PATH)

    total = len(df)
    print(f"Total reference utterances: {total}")

    print("\nPer-character counts:")
    counts = df["character"].value_counts()

    for character, count in counts.items():
        print(f"  {character}: {count}")

    print("==============================\n")


# =================================================
# Simulation Stats
# =================================================

def simulation_stats():

    print("\n==============================")
    print("SIMULATION DATA STATISTICS")
    print("==============================")

    files = glob.glob(DATA_GLOB)

    if not files:
        print("No simulation files found.")
        return

    total_utterances = 0
    total_runs = len(files)

    speaker_counts = {}
    case_counts = {}

    for path in files:
        df = pd.read_csv(path)

        total_utterances += len(df)

        # Count per speaker
        for speaker, count in df["speaker"].value_counts().items():
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + count

        # Extract case name from path
        parts = path.split(os.sep)
        # expected structure: data/caseX/run_*/dialogue_log.csv
        case_name = parts[1]
        case_counts[case_name] = case_counts.get(case_name, 0) + len(df)

    print(f"Total simulation utterances: {total_utterances}")
    print(f"Total simulation runs: {total_runs}")

    print("\nPer-character utterances (simulation):")
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count}")

    print("\nPer-case utterances:")
    for case, count in case_counts.items():
        print(f"  {case}: {count}")

    print("==============================\n")


# =================================================
# Main
# =================================================

def main():
    reference_stats()
    simulation_stats()


if __name__ == "__main__":
    main()