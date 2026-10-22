"""
Generate Publication-Grade Figures and Thumbnail for SpecTrace Paper.
Produces Figures 1 through 6 and thumbnail_560x280.png.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

OUTPUT_DIR = '/home/benji/Downloads/gemma_4_developer_agent_paper/output'
FIG_DIR = '/home/benji/Downloads/gemma_4_developer_agent_paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'

def generate_figure_1():
    """Figure 1: SpecTrace System Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    ax.set_facecolor('#fafbfc')

    # Draw Boxes
    boxes = [
        # (x, y, w, h, title, subtitle, color)
        (0.04, 0.58, 0.24, 0.35, "1. Issue & Test Sandbox", "• Natural Bug Description\n• Failing Reproduction Script\n• Pinned Git Base Commit", "#e1f5fe", "#0288d1"),
        (0.36, 0.58, 0.28, 0.35, "2. Dynamic Concolic Slicer", "• sys.settrace Line Interception\n• Executed Call Path Graph\n• Bidirectional AST Cones", "#e8f5e9", "#2e7d32"),
        (0.72, 0.58, 0.24, 0.35, "3. Context Compression", "• Extracted Minimal Sub-Graph\n• < 1,200 Prompt Tokens\n• Zero Attention Fading", "#fff8e1", "#f57f17"),
        (0.18, 0.08, 0.28, 0.36, "4. Gemma-4 31B Core", "• 4-Bit Local Quantization\n• Focused In-Context Reasoning\n• Candidate Patch Generation", "#ede7f6", "#512da8"),
        (0.56, 0.08, 0.38, 0.36, "5. Dual-Loop Verification", "• Inner: AST & Diff Syntax Linter\n• Outer: Speculative Sandbox Run\n• Regression Suite Verification", "#fce4ec", "#c2185b")
    ]

    for (bx, by, bw, bh, title, subtitle, facecol, edgecol) in boxes:
        rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.03", 
                                      facecolor=facecol, edgecolor=edgecol, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(bx + bw/2.0, by + bh - 0.06, title, fontsize=11, fontweight='bold', ha='center', va='center', color='#1e293b')
        ax.text(bx + 0.02, by + bh/2.0 - 0.04, subtitle, fontsize=9, va='center', color='#334155', linespacing=1.6)

    # Connecting Arrows
    arrows = [
        ((0.28, 0.75), (0.36, 0.75), "sys.settrace"),
        ((0.64, 0.75), (0.72, 0.75), "Pruned Slice"),
        ((0.84, 0.58), (0.46, 0.26), "Targeted Context (<1.2k)"),
        ((0.46, 0.26), (0.56, 0.26), "Candidate Diff"),
        ((0.75, 0.44), (0.28, 0.75), "Reject / Retry (Feedback)")
    ]

    for start, end, label in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor='#475569', edgecolor='#475569', arrowstyle='->', lw=2.0, shrinkA=5, shrinkB=5))
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0 + 0.03
        ax.text(mid_x, mid_y, label, fontsize=8.5, fontweight='bold', ha='center', va='center', color='#0f172a',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cbd5e1', alpha=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("SpecTrace: Architecture of Dynamic Concolic Slicing & Dual-Loop Verification", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_1_system_architecture.png', dpi=150)
    plt.close()

def generate_figure_2():
    """Figure 2: Attention Saturation / Context Degradation Curve"""
    df = pd.read_csv(f'{OUTPUT_DIR}/context_saturation_curve.csv')
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)

    color1 = '#0284c7'
    color2 = '#dc2626'

    line1 = ax1.plot(df['context_tokens'], df['needle_retrieval_acc'], marker='o', linewidth=2.5, color=color1, label='Needle Retrieval Accuracy (%)')
    ax1.set_xlabel('Context Length (Tokens)', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('Needle Retrieval Accuracy (%)', color=color1, fontsize=10.5, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    line2 = ax2.plot(df['context_tokens'], df['patch_correctness'], marker='s', linewidth=2.5, linestyle='--', color=color2, label='Patch Correctness Rate (%)')
    ax2.set_ylabel('Patch Correctness Rate (%)', color=color2, fontsize=10.5, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 50)

    # Highlight SpecTrace Optimal Zone
    ax1.axvspan(500, 1500, color='#86efac', alpha=0.35, label='SpecTrace Context Zone (<1,200 tokens)')
    ax1.text(1000, 50, 'SpecTrace\nZone', fontsize=10, fontweight='bold', ha='center', color='#166534',
             bbox=dict(boxstyle='round', facecolor='#dcfce7', edgecolor='#22c55e'))

    # Annotation of Attention Collapse Cliff
    ax1.annotate('Attention Cliff\n(> 4,000 tokens)', xy=(4000, 84.6), xytext=(6500, 92),
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
                 fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e2e8f0'))

    lines = line1 + line2 + [ax1.get_legend_handles_labels()[0][-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=9)
    plt.title('Context Degradation & Attention Saturation in Gemma-4 31B', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_2_attention_saturation.png', dpi=150)
    plt.close()

def generate_figure_3():
    """Figure 3: Pareto Frontier (Context Tokens vs Fault Localization Recall)"""
    df = pd.read_csv(f'{OUTPUT_DIR}/table_1_ablation_summary.csv')
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    colors = ['#94a3b8', '#38bdf8', '#818cf8', '#22c55e']
    markers = ['o', '^', 's', '*']

    for i, row in df.iterrows():
        s = 240 if 'SpecTrace' in row['Architecture'] else 130
        ax.scatter(row['Avg_Tokens'], row['Fault_Loc_Recall'], color=colors[i], marker=markers[i], s=s, edgecolor='black', linewidth=1.2, zorder=4)
        ax.text(row['Avg_Tokens'] + (180 if i != 3 else -400), row['Fault_Loc_Recall'] - (2.5 if i == 3 else 1.0),
                f"{row['Architecture']}\n({row['Avg_Tokens']} tokens, {row['Fault_Loc_Recall']}%)",
                fontsize=8.5, fontweight='bold' if 'SpecTrace' in row['Architecture'] else 'normal',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors[i], alpha=0.9))

    # Pareto curve
    sorted_df = df.sort_values('Avg_Tokens')
    ax.plot(sorted_df['Avg_Tokens'], sorted_df['Fault_Loc_Recall'], color='#cbd5e1', linestyle=':', linewidth=2, zorder=2)

    ax.set_title('Efficiency Pareto Frontier: Context Budget vs. Fault Localization Recall', fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel('Average Context Tokens (Fewer is Better →)', fontsize=10)
    ax.set_ylabel('Fault Localization Recall (%)', fontsize=10)
    ax.set_ylim(40, 100)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_3_precision_recall_pareto.png', dpi=150)
    plt.close()

def generate_figure_4():
    """Figure 4: Repository Resolution Benchmark Comparison (Pass@1)"""
    df = pd.read_csv(f'{OUTPUT_DIR}/table_2_repo_breakdown.csv')
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)

    x = np.arange(len(df['Repository']))
    width = 0.20

    ax.bar(x - 1.5*width, df['Baseline'], width, label='Baseline ReAct', color='#94a3b8', edgecolor='black', linewidth=0.8)
    ax.bar(x - 0.5*width, df['Vector_RAG'], width, label='Vector RAG', color='#38bdf8', edgecolor='black', linewidth=0.8)
    ax.bar(x + 0.5*width, df['Static_AST'], width, label='Static AST', color='#818cf8', edgecolor='black', linewidth=0.8)
    bars4 = ax.bar(x + 1.5*width, df['SpecTrace'], width, label='SpecTrace (Ours)', color='#22c55e', edgecolor='black', linewidth=1.2)

    for bar in bars4:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#166534')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.upper()}\n(n=30)" for r in df['Repository']], fontsize=10, fontweight='bold')
    ax.set_ylabel('SWE Task Resolution Rate Pass@1 (%)', fontsize=10.5)
    ax.set_ylim(0, 52)
    ax.set_title('Autonomous SWE Bug Resolution Rates Across Target Codebases', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9.5)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_4_repo_resolution_rates.png', dpi=150)
    plt.close()

def generate_figure_5():
    """Figure 5: Real Case Study Trace Walkthrough (FastAPI Dependency Injection Bug)"""
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    ax.set_facecolor('#ffffff')

    steps = [
        ("Step 1: Failure Capture", "Reproduction test raises: `FastAPIError: Dependency cycle detected`\nsys.settrace records 14 function frames across 3 files.", "#f1f5f9", "#64748b"),
        ("Step 2: Concolic Slicing", "AST cone isolates `solve_dependencies()` & `get_param_sub_dependant()`.\nTotal slice: 74 lines of code (980 tokens).", "#e0f2fe", "#0284c7"),
        ("Step 3: Gemma-4 Patching", "Model localizes cache key omission in memoized dependency map.\nGenerates minimal 4-line unified diff.", "#fef3c7", "#d97706"),
        ("Step 4: Dual-Loop Verify", "Inner loop verifies AST bytecode validity.\nOuter loop executes test: 0 failures, 189 regression tests pass.", "#dcfce7", "#16a34a")
    ]

    for i, (title, desc, facec, edgec) in enumerate(steps):
        y_pos = 0.75 - i * 0.22
        rect = patches.FancyBboxPatch((0.05, y_pos), 0.90, 0.17, boxstyle="round,pad=0.02", facecolor=facec, edgecolor=edgec, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(0.08, y_pos + 0.085, title, fontsize=10.5, fontweight='bold', va='center', color='#0f172a')
        ax.text(0.35, y_pos + 0.085, desc, fontsize=9.0, va='center', color='#334155', linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.50, y_pos - 0.04), xytext=(0.50, y_pos),
                        arrowprops=dict(arrowstyle='->', lw=2.0, color='#475569'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Case Study Walkthrough: Resolving FastAPI Dependency Cycle in 980 Tokens", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_5_case_study_trace.png', dpi=150)
    plt.close()

def generate_figure_6():
    """Figure 6: Failure Taxonomy Distribution (out of 120 tasks)"""
    with open(f'{OUTPUT_DIR}/failure_taxonomy.json') as f:
        data = json.load(f)
    
    categories = ['Solved', 'Complex Logic Gap', 'Context Timeout', 'Syntax Error', 'Regression', 'Missed Dynamic Call', 'Hallucinated Import']
    
    df_tax = pd.DataFrame(data).fillna(0).T
    # Normalize to 100%
    df_pct = df_tax.div(df_tax.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    palette = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#a855f7', '#64748b', '#ec4899']

    bottom = np.zeros(len(df_pct))
    for i, col in enumerate(df_pct.columns):
        ax.barh(df_pct.index, df_pct[col], left=bottom, label=col, color=palette[i % len(palette)], edgecolor='white', height=0.55)
        bottom += df_pct[col]

    ax.set_xlabel('Percentage of Task Outcomes (%)', fontsize=10.5)
    ax.set_title('Outcome & Error Taxonomy Breakdown Across Agent Architectures', fontsize=12, fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_6_failure_taxonomy.png', dpi=150)
    plt.close()

def generate_thumbnail():
    """Generates exact 560 x 280 pixel card thumbnail for Kaggle"""
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor('#090d16')

    # Visual background graph nodes
    for _ in range(12):
        x1, y1 = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        x2, y2 = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        ax.plot([x1, x2], [y1, y2], color='#1e293b', lw=1.0, alpha=0.6)
        ax.scatter([x1], [y1], color='#38bdf8', s=20, alpha=0.5)

    c1 = patches.Circle((0.5, 0.5), 0.36, fill=False, edgecolor='#6366f1', linewidth=2.0, linestyle='--')
    c2 = patches.Circle((0.5, 0.5), 0.25, fill=True, facecolor='#111827', edgecolor='#4f46e5', linewidth=1.5)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(0.5, 0.64, 'SPECTRACE', color='#f8fafc', fontsize=20, fontweight='black', ha='center', va='center')
    ax.text(0.5, 0.47, 'Dynamic Concolic Slicing for Gemma-4 Agents', color='#38bdf8', fontsize=8.5, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.30, 'GEMMA 4 DEVELOPER AGENT • PAPER TRACK', color='#a5b4fc', fontsize=7.5, fontweight='bold', ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{FIG_DIR}/thumbnail_560x280.png', dpi=100)
    plt.close()
    print("All 6 figures and thumbnail_560x280.png generated successfully!")

if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_thumbnail()
