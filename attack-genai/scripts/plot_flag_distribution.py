import json
import matplotlib.pyplot as plt
import argparse
import re
from collections import Counter
import os

def parse_moderation_info(info_str):
    """
    Parses strings like: "FLAGGED: harassment(0.8500), illicit(0.4000)"
    Returns a list of categories like ['harassment', 'illicit']
    """
    if not info_str:
        return []
    
    if "FLAGGED:" in info_str:
        # Remove prefix
        content = info_str.split("FLAGGED:", 1)[1].strip()
        # Split by comma
        parts = content.split(",")
        categories = []
        for p in parts:
            p = p.strip()
            # p like "harassment(0.8500)"
            # Extract name before first (
            match = re.match(r"([a-zA-Z0-9_]+)\(", p)
            if match:
                categories.append(match.group(1))
        return categories
    elif "SKIPPED" in info_str:
         return ["SKIPPED"]
    elif "FAILED" in info_str:
         return ["FAILED"]
    else:
         return []

def main():
    parser = argparse.ArgumentParser(description="Plot distribution of flag reasons from evaluation JSON logs.")
    parser.add_argument("json_file", type=str, help="Path to the JSON log file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output image (default: derived from json filename)")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} does not exist.")
        return

    # Determine output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_flag_dist.png"

    with open(args.json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    all_flags = []
    total_flagged_samples = 0
    
    for item in data:
        info = item.get("moderation_info", "")
        flags = parse_moderation_info(info)
        if flags and "SKIPPED" not in flags and "FAILED" not in flags:
            total_flagged_samples += 1
        all_flags.extend(flags)

    counts = Counter(all_flags)
    
    if not counts:
        print("No flags found to plot.")
        return

    # Sort items for consistent plotting
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    print(f"Total processed: {len(data)}")
    print(f"Total flagged samples: {total_flagged_samples}")
    print("Counts:")
    for l, v in zip(labels, values):
        print(f"  {l}: {v}")

    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color='skyblue', edgecolor='black')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xlabel('Flag Categories')
    plt.ylabel('Count')
    plt.title(f'Distribution of Flag Reasons\n({os.path.basename(args.json_file)})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
