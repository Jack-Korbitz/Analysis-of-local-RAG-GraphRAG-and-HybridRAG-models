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

# Match graph visualization palette (Lockheed Martin)
APPROACH_COLORS = {
    "Baseline":  "#001F5B",   # deep navy
    "Oracle":    "#4A90D9",   # cornflower blue
    "RAG":       "#7FB3E0",   # light sky blue
    "GraphRAG":  "#5A5A5A",   # dark grey
    "HybridRAG": "#B0B0B0",   # light silver grey
}

MODEL_COLORS = {
    "llama3.1:8b": "#003087",
    "gemma3:12b":  "#009CDE",
    "qwen3:8b":    "#6D6E71",
}


BG_COLOR  = "#ffffff"
AX_COLOR  = "#ffffff"
GRID_COLOR = "#dddddd"


def load_results():
    files = {
        "Baseline": METRICS_DIR / "baseline.json",
        "Oracle":   METRICS_DIR / "oracle.json",
        "RAG":      METRICS_DIR / "rag.json",
        "GraphRAG": METRICS_DIR / "graphrag.json",
        "HybridRAG": METRICS_DIR / "hybrid.json",
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

    if gt_norm and re.search(r'(?<!\d)' + re.escape(gt_norm) + r'(?!\d)', answer_norm):
        return True

    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_norm)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_norm)

    if not gt_numbers or not answer_numbers:
        return False

    try:
        gt_val = float(gt_numbers[0])

        if gt_val in (0.0, 1.0):
            if gt_val == 1.0 and re.search(r'\b(yes|true|did|exceeded?|greater|more|higher)\b', answer_str):
                return True
            if gt_val == 0.0 and re.search(r'\b(no|false|did not|not exceed|less|lower|neither)\b', answer_str):
                return True

        for ans_num in answer_numbers:
            ans_val = float(ans_num)

            if gt_val == ans_val:
                return True
            if 0 < abs(gt_val) < 1 and abs(ans_val - gt_val * 100) < 0.5:
                return True
            if 0 < abs(ans_val) < 1 and abs(ans_val * 100 - gt_val) < 0.5:
                return True
            if abs(gt_val) >= 100 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True
            if 1 <= abs(gt_val) < 100 and abs(ans_val - gt_val) < 0.5:
                return True
            if 0 < abs(gt_val) < 1 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True
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


def style_ax(ax):
    ax.set_facecolor(AX_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#333333')
    ax.yaxis.label.set_color('#333333')
    ax.xaxis.label.set_color('#333333')


def plot_accuracy_by_model_and_approach(data, ax):
    """X-axis = Approach, grouped bars = Models. Shows both dimensions clearly."""
    approaches = list(data.keys())
    present = {k.rsplit('_', 1)[0] for v in data.values() for k in v}
    all_models = [m for m in MODEL_COLORS if m in present]

    n_models = len(all_models)
    width = 0.22
    x = np.arange(len(approaches))
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, model in enumerate(all_models):
        accs = []
        for approach in approaches:
            stats = get_model_stats(data[approach])
            if model in stats and stats[model]['total'] > 0:
                accs.append(stats[model]['correct'] / stats[model]['total'] * 100)
            else:
                accs.append(0)

        bars = ax.bar(x + offsets[j], accs, width,
                      label=model,
                      color=MODEL_COLORS[model],
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11, fontweight='bold', color='#333333')
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.15))
    style_ax(ax)


def plot_accuracy_by_dataset(data, ax):
    approaches = list(data.keys())
    datasets = sorted({k.rsplit('_', 1)[1] for v in data.values() for k in v})

    x = np.arange(len(datasets))
    width = 0.15
    offsets = np.linspace(-(len(approaches) - 1) / 2, (len(approaches) - 1) / 2, len(approaches)) * width

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
                      color=APPROACH_COLORS[approach], edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.15))
    style_ax(ax)


def plot_latency(data, ax):
    approaches = list(data.keys())
    present = {k.rsplit('_', 1)[0] for v in data.values() for k in v}
    all_models = [m for m in MODEL_COLORS if m in present]

    n_models = len(all_models)
    width = 0.22
    x = np.arange(len(approaches))
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, model in enumerate(all_models):
        lats = []
        for approach in approaches:
            stats = get_model_stats(data[approach])
            if model in stats and stats[model]['latency']:
                lats.append(sum(stats[model]['latency']) / len(stats[model]['latency']) / 1000)
            else:
                lats.append(0)

        bars = ax.bar(x + offsets[j], lats, width,
                      label=model,
                      color=MODEL_COLORS[model],
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, lats):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_ylabel('Latency (seconds)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11, fontweight='bold')
    style_ax(ax)


def plot_overall_summary(data, ax):
    """Overall accuracy per approach, with per-model breakdown as stacked annotations."""
    approaches = list(data.keys())
    present = {k.rsplit('_', 1)[0] for v in data.values() for k in v}
    all_models = [m for m in MODEL_COLORS if m in present]

    overall = {}
    for approach in approaches:
        all_q = [q for qs in data[approach].values() for q in qs]
        correct = sum(check_accuracy(q['answer'], q['ground_truth']) for q in all_q)
        overall[approach] = correct / len(all_q) * 100 if all_q else 0

    bars = ax.bar(approaches, [overall[a] for a in approaches],
                  color=[APPROACH_COLORS[a] for a in approaches],
                  edgecolor='white', linewidth=0.8, width=0.5)

    for bar, approach in zip(bars, approaches):
        val = overall[approach]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#333333')
        # Per-model breakdown inside bar
        stats = get_model_stats(data[approach])
        lines = []
        for m in all_models:
            if m in stats and stats[m]['total'] > 0:
                acc = stats[m]['correct'] / stats[m]['total'] * 100
                lines.append(f"{m.split(':')[0]}: {acc:.1f}%")
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                "\n".join(lines), ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold',
                multialignment='center')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_ylim(0, max(60, max(overall.values()) * 1.25))
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(approaches, fontsize=12, fontweight='bold')
    style_ax(ax)


def strict_em(answer, gt):
    a = re.sub(r'(\d),(\d)', r'\1\2', str(answer).strip().lower())
    g = re.sub(r'(\d),(\d)', r'\1\2', re.sub(r'\.0$', '', str(gt).strip().lower()))
    return g != '' and g in a


def plot_strict_vs_relaxed(data, ax):
    """Grouped bars: Relaxed accuracy vs Strict EM per approach."""
    approaches = list(data.keys())
    relaxed, strict = [], []
    for approach in approaches:
        all_q = [q for qs in data[approach].values() for q in qs]
        n = len(all_q) or 1
        r = sum(check_accuracy(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        s = sum(strict_em(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        relaxed.append(r)
        strict.append(s)

    x = np.arange(len(approaches))
    width = 0.30
    bars_r = ax.bar(x - width / 2, relaxed, width, label='Relaxed',
                    color='#4A90D9', edgecolor='white', linewidth=0.8)
    bars_s = ax.bar(x + width / 2, strict, width, label='Strict EM',
                    color='#5A5A5A', edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars_r, relaxed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    for bar, val in zip(bars_s, strict):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(60, max(relaxed) * 1.20))
    ax.legend(fontsize=10, frameon=True, edgecolor='#cccccc')
    style_ax(ax)


def plot_answer_in_context(data, ax):
    """Bar chart: answer-in-context % for approaches that have retrieval_metrics."""
    retrieval_approaches = [a for a in data if a != "Baseline"]
    aic_pcts = []
    for approach in retrieval_approaches:
        all_q = [q for qs in data[approach].values() for q in qs]
        rm_qs = [q for q in all_q if q.get('retrieval_metrics')]
        if rm_qs:
            aic = sum(1 for q in rm_qs if q['retrieval_metrics'].get('answer_in_context')) / len(rm_qs) * 100
        else:
            aic = 0
        aic_pcts.append(aic)

    x = np.arange(len(retrieval_approaches))
    bars = ax.bar(x, aic_pcts, 0.5,
                  color=[APPROACH_COLORS[a] for a in retrieval_approaches],
                  edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, aic_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')

    ax.set_ylabel('Answer-in-Context (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(retrieval_approaches, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(40, max(aic_pcts) * 1.25) if aic_pcts else 40)
    style_ax(ax)


def plot_conditional_accuracy(data, ax):
    """Grouped bars: overall vs when-in-context vs when-not-in-context accuracy."""
    retrieval_approaches = [a for a in data if a != "Baseline"]
    overall, cond_in, cond_out = [], [], []
    for approach in retrieval_approaches:
        all_q = [q for qs in data[approach].values() for q in qs]
        n = len(all_q) or 1
        ov = sum(check_accuracy(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        aic_qs = [q for q in all_q if q.get('retrieval_metrics', {}).get('answer_in_context')]
        non_qs = [q for q in all_q if not q.get('retrieval_metrics', {}).get('answer_in_context')]
        ci = sum(check_accuracy(q['answer'], q['ground_truth']) for q in aic_qs) / len(aic_qs) * 100 if aic_qs else 0
        co = sum(check_accuracy(q['answer'], q['ground_truth']) for q in non_qs) / len(non_qs) * 100 if non_qs else 0
        overall.append(ov)
        cond_in.append(ci)
        cond_out.append(co)

    x = np.arange(len(retrieval_approaches))
    width = 0.22
    bars1 = ax.bar(x - width, overall, width, label='Overall', color='#001F5B', edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, cond_in, width, label='Answer in context', color='#4A90D9', edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + width, cond_out, width, label='Answer NOT in context', color='#B0B0B0', edgecolor='white', linewidth=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            val = bar.get_height()
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(retrieval_approaches, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(60, max(cond_in + overall) * 1.20) if cond_in else 60)
    ax.legend(fontsize=9, frameon=True, edgecolor='#cccccc')
    style_ax(ax)


def plot_graphrag_strategy_dist(data, ax):
    """Pie chart: GraphRAG retrieval strategy distribution."""
    if 'GraphRAG' not in data:
        ax.text(0.5, 0.5, 'No GraphRAG data', ha='center', va='center', transform=ax.transAxes)
        return
    strats = {}
    for qs in data['GraphRAG'].values():
        for q in qs:
            s = q.get('retrieval_metrics', {}).get('retrieval_strategy', 'unknown')
            strats[s] = strats.get(s, 0) + 1
    if not strats:
        ax.text(0.5, 0.5, 'No strategy data', ha='center', va='center', transform=ax.transAxes)
        return

    sorted_strats = sorted(strats.items(), key=lambda x: -x[1])
    labels = [s[0] for s in sorted_strats]
    sizes = [s[1] for s in sorted_strats]
    blues_greys = ['#001F5B', '#4A90D9', '#7FB3E0', '#5A5A5A', '#B0B0B0', '#D0D0D0']
    colors = blues_greys[:len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.0f%%', startangle=140,
        colors=colors, textprops={'fontsize': 10, 'color': '#333333'})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=9, frameon=True, edgecolor='#cccccc')


def plot_graphrag_stratified_accuracy(data, ax):
    """Bar chart: accuracy per GraphRAG retrieval strategy."""
    if 'GraphRAG' not in data:
        ax.text(0.5, 0.5, 'No GraphRAG data', ha='center', va='center', transform=ax.transAxes)
        return
    strat_acc = {}
    for qs in data['GraphRAG'].values():
        for q in qs:
            s = q.get('retrieval_metrics', {}).get('retrieval_strategy', 'unknown')
            if s not in strat_acc:
                strat_acc[s] = {'correct': 0, 'total': 0}
            strat_acc[s]['total'] += 1
            if check_accuracy(q['answer'], q['ground_truth']):
                strat_acc[s]['correct'] += 1

    sorted_strats = sorted(strat_acc.items(), key=lambda x: -x[1]['total'])
    labels = [s[0] for s in sorted_strats]
    accs = [s[1]['correct'] / s[1]['total'] * 100 if s[1]['total'] else 0 for s in sorted_strats]
    counts = [s[1]['total'] for s in sorted_strats]
    blues_greys = ['#001F5B', '#4A90D9', '#7FB3E0', '#5A5A5A', '#B0B0B0', '#D0D0D0']
    colors = blues_greys[:len(labels)]

    x = np.arange(len(labels))
    bars = ax.bar(x, accs, 0.5, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy (%)', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold', rotation=20, ha='right')
    ax.set_ylim(0, max(60, max(accs) * 1.25) if accs else 60)
    style_ax(ax)


def add_approach_legend(fig, approaches):
    handles = [mpatches.Patch(color=APPROACH_COLORS[a], label=a) for a in approaches]
    fig.legend(handles=handles, loc='lower center', ncol=len(approaches),
               fontsize=10, frameon=True, framealpha=1.0,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.01))


def add_model_legend(fig):
    handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m)
        for m in ["llama3.1:8b", "gemma3:12b", "qwen3:8b"]
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=11, frameon=True, framealpha=1.0,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.01))


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

    # Charts grouped by approach (model bars within each approach)
    model_charts = [
        ("accuracy_by_model.png",  "Accuracy by Model and Approach",    plot_accuracy_by_model_and_approach),
        ("latency_by_model.png",   "Avg Latency per Question by Model", plot_latency),
    ]
    for filename, title, plot_fn in model_charts:
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle(title, fontsize=14, fontweight='bold', color='#333333', y=1.01)
        plot_fn(data, ax)
        add_model_legend(fig)
        save_figure(fig, OUTPUT_DIR / filename)

    # Charts grouped by approach color
    approach_charts = [
        ("overall_accuracy.png",    "Overall Accuracy by Approach",     plot_overall_summary),
        ("accuracy_by_dataset.png", "Accuracy by Dataset and Approach", plot_accuracy_by_dataset),
    ]
    approaches = list(data.keys())
    for filename, title, plot_fn in approach_charts:
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle(title, fontsize=14, fontweight='bold', color='#333333', y=1.01)
        plot_fn(data, ax)
        add_approach_legend(fig, approaches)
        save_figure(fig, OUTPUT_DIR / filename)

    # Strict EM vs Relaxed
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Relaxed vs Strict Exact Match Accuracy', fontsize=14, fontweight='bold', color='#333333', y=1.01)
    plot_strict_vs_relaxed(data, ax)
    save_figure(fig, OUTPUT_DIR / "strict_vs_relaxed.png")

    # Answer-in-context
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Answer-in-Context Rate by Approach', fontsize=14, fontweight='bold', color='#333333', y=1.01)
    plot_answer_in_context(data, ax)
    save_figure(fig, OUTPUT_DIR / "answer_in_context.png")

    # Generator conditional accuracy
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Generator Conditional Accuracy', fontsize=14, fontweight='bold', color='#333333', y=1.01)
    plot_conditional_accuracy(data, ax)
    save_figure(fig, OUTPUT_DIR / "conditional_accuracy.png")

    # GraphRAG strategy distribution (pie)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('GraphRAG Retrieval Strategy Distribution', fontsize=14, fontweight='bold', color='#333333', y=1.01)
    plot_graphrag_strategy_dist(data, ax)
    save_figure(fig, OUTPUT_DIR / "graphrag_strategy_dist.png")

    # GraphRAG stratified accuracy
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('GraphRAG Accuracy by Retrieval Strategy', fontsize=14, fontweight='bold', color='#333333', y=1.01)
    plot_graphrag_stratified_accuracy(data, ax)
    save_figure(fig, OUTPUT_DIR / "graphrag_stratified_accuracy.png")


if __name__ == "__main__":
    main()
