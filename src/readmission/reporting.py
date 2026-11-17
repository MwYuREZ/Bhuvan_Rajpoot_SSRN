"""Publication-style static figures and an evidence-linked Chinese run report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import calibration_deciles

COLORS = {"admission": "#365f82", "discharge": "#d3484a"}
LABELS = {"admission": "Admission", "discharge": "Discharge"}


def write_figures(out: Path, y, predictions, intervals):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130,
                         "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for name, p in predictions.items():
        stats = intervals[intervals.features.eq(name)].set_index("metric")
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=COLORS[name],
                     label=f"{LABELS[name]} (AUC {stats.loc['auroc', 'estimate']:.3f})")
        axes[1].plot(recall, precision, color=COLORS[name],
                     label=f"{LABELS[name]} (AP {stats.loc['ap', 'estimate']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].axhline(np.mean(y), linestyle="--", color="gray", linewidth=1,
                   label=f"Prevalence {np.mean(y):.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=9)
    fig.savefig(figures / "roc_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), layout="constrained")
    all_bins = []
    for name, p in predictions.items():
        bins = calibration_deciles(y, p)
        bins.insert(0, "features", name)
        all_bins.append(bins)
        ax.plot(bins.mean_predicted, bins.event_rate, "o-", color=COLORS[name],
                label=LABELS[name], markersize=4)
    pd.concat(all_bins).to_csv(out / "tables" / "calibration_deciles.csv", index=False)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlim=(0, .45), ylim=(0, .45), xlabel="Mean predicted probability",
           ylabel="Observed event proportion", title="Held-out calibration by probability decile")
    ax.legend()
    ax.grid(alpha=.15)
    fig.savefig(figures / "calibration.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, metric, title in zip(axes, ["auroc", "ap", "brier"],
                                 ["AUROC", "Average precision", "Brier (lower is better)"]):
        for i, name in enumerate(["admission", "discharge"]):
            row = intervals.loc[intervals.features.eq(name) & intervals.metric.eq(metric)].iloc[0]
            ax.errorbar(row.estimate, i, xerr=[[max(0, row.estimate - row.lower)],
                         [max(0, row.upper - row.estimate)]], fmt="o", capsize=4, color=COLORS[name])
        ax.set(yticks=[0, 1], yticklabels=["Admission", "Discharge"], ylim=(-.5, 1.5), title=title)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Selected models: 95% patient-cluster bootstrap intervals")
    fig.savefig(figures / "performance_intervals.png")
    plt.close(fig)


def paper_comparison(out: Path, candidates, operating, intervals, reference):
    refs = pd.DataFrame(reference["candidates"])
    comparison = candidates.merge(refs, on=["features", "model"], suffixes=("_reproduction", "_paper"))
    for metric in ["dev_auroc", "dev_ap", "test_auroc", "test_ap", "test_brier", "test_calibration_slope"]:
        comparison[metric + "_difference"] = comparison[metric + "_reproduction"] - comparison[metric + "_paper"]
    comparison.to_csv(out / "tables" / "paper_vs_reproduction.csv", index=False)
    operating.merge(pd.DataFrame(reference["thresholds"]), on=["features", "criterion"],
                    suffixes=("_reproduction", "_paper")).to_csv(
        out / "tables" / "thresholds_vs_paper.csv", index=False)
    intervals.merge(pd.DataFrame(reference["selected_intervals"]), on=["features", "metric"],
                    suffixes=("_reproduction", "_paper")).to_csv(
        out / "tables" / "intervals_vs_paper.csv", index=False)
    return comparison


def write_report(out, candidates, operating, intervals, cohort_summary, selection, runtime, quick=False):
    def interval(name, metric):
        return intervals.loc[intervals.features.eq(name) & intervals.metric.eq(metric)].iloc[0]

    lines = ["# Rajpoot (SSRN 7154739) 獨立復現結果", "",
             "這是依論文文字重建的方法，不是作者未公開程式碼的精確重跑。",
             f"執行時間（UTC）：{runtime['started_utc']}；模式：{'快速流程檢查，不能作為論文復現' if quick else '完整資料、完整模型與 500 次病人 bootstrap' if runtime['config']['bootstrap_replicates'] == 500 else '完整資料、自訂 bootstrap 次數'}。", "",
             "## 資料與切分", "",
             f"來源 CSV SHA-256：`{runtime['source_sha256']}`，與論文一致。",
             "病人互斥檢查已通過；開發集每筆紀錄恰好有一次 OOF 預測。", "",
             "| 分區 | 住院紀錄 | 病人 | 陽性 | 盛行率 |", "|---|---:|---:|---:|---:|"]
    for name, row in cohort_summary.items():
        lines.append(f"| {name} | {row['encounters']:,} | {row['patients']:,} | {row['positives']:,} | {row['prevalence']:.4%} |")
    lines += ["", "## 模型選擇及測試表現", "",
              "模型由開發集 OOF AP 最大者選出，Brier 最低者打破平手；選擇與 F1/F2 閾值在測試評估前寫入 `selection_lock.json`。", "",
              "| 特徵 | 模型 | 選出 | Dev AP | Test AUROC | Test AP | Brier | 校準截距 | 校準斜率 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in candidates.iterrows():
        lines.append(f"| {r.features} | {r.model} | {'是' if r.selected else ''} | {r.dev_ap:.4f} | {r.test_auroc:.4f} | {r.test_ap:.4f} | {r.test_brier:.5f} | {r.test_calibration_intercept:.3f} | {r.test_calibration_slope:.3f} |")
    lines += ["", "## 入院與出院資訊的配對差異", "",
              "以下一律為出院減入院；兩模型使用相同的病人 bootstrap 抽樣，並保留被抽中病人的全部住院紀錄。", "",
              "| 指標 | 差異 | 95% 百分位區間 | 論文差異 |", "|---|---:|---|---:|"]
    for metric, paper in [("auroc", .046), ("ap", .032), ("brier", -.0022)]:
        r = interval("difference", metric)
        lines.append(f"| {metric} | {r.estimate:.5f} | [{r.lower:.5f}, {r.upper:.5f}] | {paper:.4f} |")
    lines += ["", "## 開發集選定的操作閾值", "",
              "| 特徵 | 準則 | 閾值 | 敏感度 | 特異度 | PPV | NPV | F1 | F2 | 警示比例 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in operating.iterrows():
        lines.append(f"| {r.features} | {r.criterion} | {r.threshold:.5f} | {r.sensitivity:.3f} | {r.specificity:.3f} | {r.ppv:.3f} | {r.npv:.3f} | {r.f1:.3f} | {r.f2:.3f} | {r.alert_rate:.1%} |")
    a = interval("admission", "auroc")
    d = interval("discharge", "auroc")
    delta = interval("difference", "auroc")
    ap_delta = interval("difference", "ap")
    lines += ["", "## 與論文的一致程度", "",
              f"本次入院 AUROC 為 {a.estimate:.4f}（論文 0.631），出院為 {d.estimate:.4f}（論文 0.677）。",
              f"配對 AUROC 差異為 {delta.estimate:.4f}，AP 差異為 {ap_delta.estimate:.4f}。",
              ("AUROC 與 AP 差異的區間皆大於零，支持論文『出院資訊提供有限但明確的改善』的方向性結論。"
               if delta.lower > 0 and ap_delta.lower > 0 else "至少一項差異區間涵蓋零，不能重述論文的統計明確性結論。"),
              f"選出的模型為入院 {selection['admission']['model']}、出院 {selection['discharge']['model']}；論文為 logistic、hgb。",
              "每個候選模型的逐項誤差見 `tables/paper_vs_reproduction.csv`；閾值與區間也有對照表。",
              "不能把接近的數值當作已找回作者實作。未公開的診斷映射、類別處理與版本可能影響差異，且本次未依測試結果搜尋這些選擇。", "",
              "## 圖與可稽核輸出", "",
              "![ROC and PR](figures/roc_pr.png)", "", "![Calibration](figures/calibration.png)", "",
              "![Patient-cluster intervals](figures/performance_intervals.png)", "",
              "- `predictions/development_oof.csv.gz`：四個候選模型的 OOF 預測。",
              "- `predictions/test_predictions.csv.gz`：四個候選模型的測試預測。",
              "- `splits/encounter_assignments.csv.gz`：原始列、病人、分區與 CV 折。",
              "- `tables/bootstrap_replicates.csv.gz`：每次配對抽樣的指標與差異。",
              "- `tables/subgroups.csv`：至少 200 筆且包含兩類結果的性別／種族描述性分析。",
              "- `runtime.json`：參數、環境、原始資料與程式碼雜湊。",
              "- `models/`：四個訓練完成的前處理與模型 pipeline（僅載入可信任的 joblib 檔）。", "",
              "## 限制", "",
              "這是同一歷史資料來源的內部驗證；bootstrap 只估計固定預測下的病人抽樣不確定性，不涵蓋重新訓練、跨院或時間漂移。",
              "入院特徵可用性依欄位語意近似。族群分析是描述性結果，不構成公平性認證。",
              "完整方法假設與範例 repo 的差異見專案根目錄 `ASSUMPTIONS_AND_PAPER_ISSUES.md`。", ""]
    (out / "REPRODUCTION_REPORT_zh-TW.md").write_text("\n".join(lines), encoding="utf-8")
