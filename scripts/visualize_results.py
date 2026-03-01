import sys
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path("results/visualizations")
METRICS_DIR = Path("results/metrics")

COLORS = {
    "Baseline": "#6c757d",
    "RAG":      "#2196F3",
    "GraphRAG": "#4CAF50",
}

MODEL_COLORS = {
    "llama3.1:8b": "#FF6B6B",
    "gemma3:12b":  "#4ECDC4",
    "qwen3:8b":    "#FFE66D",
}


def load_results():
    files = {
        "Baseline": METRICS_DIR / "baseline_fast.json",
        "RAG":      METRICS_DIR / "rag_fast.json",
        "GraphRAG": METRICS_DIR / "graphrag_fast.json",
    }
    data = {}
    for label, path in files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[label] = json.load(f)
    return data


def check_accuracy(answer, ground_truth):
    answer_lower = str(answer).lower()
    gt_str = str(ground_truth).lower().replace('.0', '')

    if gt_str in answer_lower:
        return True

    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_lower)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_str)

    if gt_numbers and answer_numbers:
        try:
            gt_val = float(gt_numbers[0])
            for ans_num in answer_numbers:
                ans_val = float(ans_num)
                if abs(gt_val) < 1:
                    if abs(ans_val - gt_val) / max(abs(gt_val), 0.001) < 0.05:
                        return True
                else:
                    if abs(ans_val - gt_val) < 0.1:
                        return True
        except ValueError:
            pass
    return False


def get_model_stats(results_dict):
    stats = {}
    for key, questions in results_dict.items():
        model = key.rsplit('_', 1)[0]
        if model not in stats:
            stats[model] = {'correct': 0, 'total': 0, 'latency': []}
        for q in questions:
            stats[model]['correct'] += check_accuracy(q['answer'], q['ground_truth'])
            stats[model]['total'] += 1
            stats[model]['latency'].append(q['latency_ms'])
    return stats


def plot_accuracy_by_approach(data, ax):
    approaches = list(data.keys())
    all_models = sorted({k.rsplit('_', 1)[0] for v in data.values() for k in v})

    x = np.arange(len(all_models))
    width = 0.25
    offsets = np.linspace(-(len(approaches)-1)/2, (len(approaches)-1)/2, len(approaches)) * width

    for i, approach in enumerate(approaches):
        stats = get_model_stats(data[approach])
        accuracies = [
            (stats[m]['correct'] / stats[m]['total'] * 100) if m in stats and stats[m]['total'] > 0 else 0
            for m in all_models
        ]
        bars = ax.bar(x + offsets[i], accuracies, width, label=approach,
                      color=COLORS[approach], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Accuracy by Model and Approach', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(':', '\n') for m in all_models], fontsize=9)
    ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def plot_accuracy_by_dataset(data, ax):
    approaches = list(data.keys())
    datasets = sorted({k.rsplit('_', 1)[1] for v in data.values() for k in v})

    x = np.arange(len(datasets))
    width = 0.25
    offsets = np.linspace(-(len(approaches)-1)/2, (len(approaches)-1)/2, len(approaches)) * width

    for i, approach in enumerate(approaches):
        accs = []
        for ds in datasets:
            questions = []
            for key, qs in data[approach].items():
                if key.endswith(f'_{ds}'):
                    questions.extend(qs)
            if questions:
                correct = sum(check_accuracy(q['answer'], q['ground_truth']) for q in questions)
                accs.append(correct / len(questions) * 100)
            else:
                accs.append(0)

        bars = ax.bar(x + offsets[i], accs, width, label=approach,
                      color=COLORS[approach], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Accuracy by Dataset and Approach', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=9)
    ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def plot_latency(data, ax):
    approaches = list(data.keys())
    all_models = sorted({k.rsplit('_', 1)[0] for v in data.values() for k in v})

    x = np.arange(len(all_models))
    width = 0.25
    offsets = np.linspace(-(len(approaches)-1)/2, (len(approaches)-1)/2, len(approaches)) * width

    for i, approach in enumerate(approaches):
        stats = get_model_stats(data[approach])
        latencies = [
            (sum(stats[m]['latency']) / len(stats[m]['latency']) / 1000) if m in stats and stats[m]['latency'] else 0
            for m in all_models
        ]
        bars = ax.bar(x + offsets[i], latencies, width, label=approach,
                      color=COLORS[approach], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=7)

    ax.set_title('Avg Latency per Question by Model', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Latency (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(':', '\n') for m in all_models], fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def plot_overall_summary(data, ax):
    approaches = list(data.keys())
    overall = {}
    for approach in approaches:
        all_q = [q for qs in data[approach].values() for q in qs]
        correct = sum(check_accuracy(q['answer'], q['ground_truth']) for q in all_q)
        overall[approach] = correct / len(all_q) * 100 if all_q else 0

    bars = ax.bar(approaches, [overall[a] for a in approaches],
                  color=[COLORS[a] for a in approaches],
                  edgecolor='white', linewidth=0.5, width=0.5)

    for bar, approach in zip(bars, approaches):
        val = overall[approach]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Overall Accuracy by Approach', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(55, max(overall.values()) * 1.2))
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    if not data:
        print("No result files found. Run benchmarks first.")
        return

    print(f"Loaded: {list(data.keys())}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('GraphRAG Benchmark Results', fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#f8f9fa')
    for ax in axes.flat:
        ax.set_facecolor('#ffffff')

    plot_overall_summary(data, axes[0, 0])
    plot_accuracy_by_approach(data, axes[0, 1])
    plot_accuracy_by_dataset(data, axes[1, 0])
    plot_latency(data, axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUT_DIR / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
