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
    "Baseline": "#bfad8a",
    "RAG":      "#8c7b23",
    "GraphRAG": "#154001",
}

MODEL_COLORS = {
    "llama3.1:8b": "#FF6B6B",
    "gemma3:12b":  "#4ECDC4",
    "qwen3:8b":    "#FFE66D",
}


def load_results():
    files = {
        "Baseline": METRICS_DIR / "baseline.json",
        "RAG":      METRICS_DIR / "rag.json",
        "GraphRAG": METRICS_DIR / "graphrag.json",
    }
    data = {}
    for label, path in files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[label] = json.load(f)
    return data


def check_accuracy(answer, ground_truth):
    answer_str = str(answer).lower()
    gt_str = re.sub(r'\.0$', '', str(ground_truth).lower())

    answer_norm = re.sub(r'(\d),(\d)', r'\1\2', answer_str)
    gt_norm = re.sub(r'(\d),(\d)', r'\1\2', gt_str)

    # Word-boundary safe string match (avoids "531" matching "-531" or "2531")
    if gt_norm and re.search(r'(?<!\d)' + re.escape(gt_norm) + r'(?!\d)', answer_norm):
        return True

    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_norm)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_norm)

    if not gt_numbers or not answer_numbers:
        return False

    try:
        gt_val = float(gt_numbers[0])

        # Boolean yes/no: GT is 0 or 1
        if gt_val in (0.0, 1.0):
            if gt_val == 1.0 and re.search(r'\b(yes|true|did|exceeded?|greater|more|higher)\b', answer_str):
                return True
            if gt_val == 0.0 and re.search(r'\b(no|false|did not|not exceed|less|lower|neither)\b', answer_str):
                return True

        for ans_num in answer_numbers:
            ans_val = float(ans_num)

            if gt_val == ans_val:
                return True

            # Percentage ↔ decimal
            if 0 < abs(gt_val) < 1 and abs(ans_val - gt_val * 100) < 0.5:
                return True
            if 0 < abs(ans_val) < 1 and abs(ans_val * 100 - gt_val) < 0.5:
                return True

            # Relative tolerance for large numbers (within 1%)
            if abs(gt_val) >= 100 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True

            # Absolute tolerance for mid-range numbers 1–99 (within 0.5)
            if 1 <= abs(gt_val) < 100 and abs(ans_val - gt_val) < 0.5:
                return True

            # Relative tolerance for small decimals <1 (within 1%)
            if abs(gt_val) < 1 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True

            # Unit scale: model strips "million"/"thousand" suffix
            if abs(gt_val) >= 1000:
                for scale in [1_000, 1_000_000]:
                    if abs(ans_val * scale - gt_val) / abs(gt_val) < 0.01:
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
                      color=COLORS[approach], edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(':', '\n') for m in all_models], fontsize=9)
    ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
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
                      color=COLORS[approach], edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=9)
    ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
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
                      color=COLORS[approach], edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Latency (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(':', '\n') for m in all_models], fontsize=9)
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
                  edgecolor='black', linewidth=0.8, width=0.5)

    for bar, approach in zip(bars, approaches):
        val = overall[approach]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(55, max(overall.values()) * 1.2))
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def add_legend(fig):
    legend_handles = [
        mpatches.Patch(color=COLORS[a], label=a) for a in ["Baseline", "RAG", "GraphRAG"]
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=11, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))


def save_figure(fig, path):
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    if not data:
        print("No result files found. Run benchmarks first.")
        return

    print(f"Loaded: {list(data.keys())}")

    plots = [
        ("overall_accuracy.png",    "Overall Accuracy by Approach",        plot_overall_summary),
        ("accuracy_by_model.png",   "Accuracy by Model and Approach",      plot_accuracy_by_approach),
        ("accuracy_by_dataset.png", "Accuracy by Dataset and Approach",    plot_accuracy_by_dataset),
        ("latency_by_model.png",    "Avg Latency per Question by Model",   plot_latency),
    ]

    for filename, title, plot_fn in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plot_fn(data, ax)
        add_legend(fig)
        save_figure(fig, OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
