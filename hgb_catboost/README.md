# HGB＋CatBoost 延伸實驗

這裡整理兩組 UCI 糖尿病住院資料的延伸實驗。兩組都使用論文復現的 **80/20 病人互斥切分**，開發集 79,398 筆，測試集 19,942 筆（13,998 位病人、2,297 筆 30 天再入院）。主要指標是 average precision（AP）。 [原論文](https://ssrn.com/abstract=7154739) · [復現說明](../../README.md)

## 實驗與結果

### 46 欄對照

復現 HGB 與 CatBoost 使用相同的 46 個出院欄位，三個 ICD-9 診斷欄位都以診斷家族表示。兩個模型各自處理類別特徵。另測固定 50/50 的預測機率平均，以及 CatBoost 將三個診斷家族欄換回原始 ICD-9 的敏感度分析。

| 方法 | 測試 AP |
|---|---:|
| 復現 HGB | 0.237372 |
| CatBoost（診斷家族，開發集選定） | **0.251383** |
| HGB＋CatBoost（固定 50/50） | 0.249442 |
| CatBoost（原始 ICD-9，敏感度分析） | 0.252763 |

CatBoost 的深度與樹數以開發集三折 OOF AP 選擇，Brier 分數用於打破平手。選定深度 **6**、**750 棵樹**，以 CPU 重訓。診斷家族版相對復現 HGB 增加 **0.014011 AP**，病人配對 bootstrap 95% CI **[0.005947, 0.022333]**。 [選模紀錄](reference/46_fields/development_trials.csv) · [完整指標](reference/46_fields/test_metrics.csv) · [配對區間](reference/46_fields/paired_differences.csv)

### 85／88 欄融合

在原始出院欄位上加入 UCI 衍生的臨床、用藥與就醫利用特徵。HGB 共 **85 欄**，使用診斷家族；CatBoost 共 **88 欄**，同時保留診斷家族及原始診斷碼。兩模型各占 50%，以機率平均後重新計算 AP。

| 方法 | 測試 AP |
|---|---:|
| 復現 HGB（46 欄參考） | 0.237372 |
| 原生類別 HGB（85 欄） | 0.244354 |
| CatBoost（88 欄） | 0.249974 |
| HGB＋CatBoost（固定 50/50） | **0.251195** |

使用相同三折開發分組，HGB 選 **100 棵樹**；CatBoost 選深度 **6**，取各折最佳樹數的中位數 **439**，以 GPU 重訓。融合相對 88 欄 CatBoost 單模增加 **0.001221 AP**，95% CI **[−0.001838, +0.004295]**。相對復現 HGB 的 **+0.013823 AP** 涉及特徵、模型與融合的共同改動。 [選模紀錄](reference/expanded/development_trials.csv) · [完整指標](reference/expanded/test_metrics.csv) · [配對區間](reference/expanded/paired_ap.csv)

兩組模型都維持原始類別比例，使用 UCI 住院資料。兩組共用測試病人，表中的分數是這份測試集的 AP。區間以同一批病人為單位配對抽樣，描述固定模型下的測試抽樣不確定性。

## 重跑

將本資料夾放在復現 repo 的 `experiments/hgb_catboost/`。程式會讀取根目錄的 `data/raw/diabetic_data.csv` 與 `results/full/` 的病人切分、執行紀錄和原 HGB 預測。如果尚無這些檔案，先按照[原 repo 的復現步驟](../../README.md#reproduce)下載資料並執行 `reproduce.py`。

以下從 repo 根目錄執行，使用 Python 3.12。兩組使用不同的 scikit-learn 版本，分別裝在獨立環境。

```powershell
# 46 欄對照
python -m venv experiments/hgb_catboost/.venv-46
.\experiments\hgb_catboost\.venv-46\Scripts\python.exe -m pip install -r experiments/hgb_catboost/requirements-46.txt
.\experiments\hgb_catboost\.venv-46\Scripts\python.exe experiments/hgb_catboost/run.py 46_fields

# 85／88 欄融合，沿用原 GPU 設定
python -m venv experiments/hgb_catboost/.venv-expanded
.\experiments\hgb_catboost\.venv-expanded\Scripts\python.exe -m pip install -r experiments/hgb_catboost/requirements-expanded.txt
.\experiments\hgb_catboost\.venv-expanded\Scripts\python.exe experiments/hgb_catboost/run.py expanded
```

`run.py` 載入已保存的 `selection_lock.json`，用完整開發集重訓，再在原測試集評估。`reference/` 收錄原始選模紀錄、選定參數、分數與配對差值，供重跑時核對。新產物寫到 `runs/46_fields/` 或 `runs/expanded/`，包含指標、配對 AP 區間、預測、模型和版本資訊。重跑可用 `--output` 指定另一個新目錄。

85／88 欄的 CatBoost 原實驗使用 NVIDIA GPU。GPU 執行可能出現小幅差異，沒有 GPU 時可指定 `expanded --device CPU`，作為另一個運算設定。衍生欄位的程式整理自[團隊 repo 的固定版本](https://github.com/Roy4016/-diabetes-project/tree/0625563bf2362865346cb6df0f15183709189254)，授權見 [TEAM_LICENSE](TEAM_LICENSE)。
