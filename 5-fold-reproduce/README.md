# HGB 基準模型的病人互斥五折交叉驗證

本實驗以 UCI 糖尿病住院資料預測 30 天內再住院，納入 99,343 筆住院紀錄、69,990 位病人，並保留三筆 `gender=Unknown/Invalid` 紀錄。外層評估採分層分組五折交叉驗證（stratified group 5-fold cross-validation），以 `patient_nbr` 分組，設定 `shuffle=True`、`random_state=777`。每位病人只出現在一個測試折，各折測試集依序為 19,869、19,869、19,868、19,869、19,868 筆。

切分由 scikit-learn 1.8.0 的 `StratifiedGroupKFold` 產生。模型訓練使用 scikit-learn 1.5.1，並讀取固定切分清單，確保逐折比較使用相同病人。 [scikit-learn 1.8 版本說明](https://scikit-learn.org/1.8/whats_new/v1.8.html#sklearn-model-selection)

## 模型與評估

兩組模型都使用 46 個出院欄位及相同的直方圖梯度提升（Histogram Gradient Boosting, HGB）設定，共訓練 200 輪。基準模型將 ICD-9 歸入診斷章節，以年齡區間中點表示年齡，並對類別欄位做眾數補值與序數編碼。對照組保留原始 ICD-9 與年齡區間，以 `Missing` 表示類別缺值，將訓練折內出現少於 100 次的類別合併後做 one-hot 編碼。兩組的數值缺值均以中位數補值，前處理與模型只在各折的外層訓練集擬合。

平均精確率（average precision, AP）以 `average_precision_score` 計算。主要指標為五個測試折 AP 的算術平均。Pooled OOF AP 則是合併所有折外（out-of-fold, OOF）預測後計算的 AP。

| 方法 | 五折平均 AP | 折間標準差 | Pooled OOF AP |
| --- | ---: | ---: | ---: |
| HGB 基準模型 | **0.228703** | 0.002883 | 0.227821 |
| HGB（原始 ICD-9 與 one-hot 編碼） | 0.237261 | 0.001932 | 0.235940 |

在相同切分與 HGB 參數下，對照組的五折平均 AP 高出 0.008558。

## 後續實驗沿用切分

- [切分清單](results/five_fold/fold_assignments.csv.gz) 以 `encounter_id` 識別資料列。第 `k` 折以 `Fold == k` 作為測試集，其餘為外層訓練集。若需要內層驗證，`inner_val_k == 1` 標示外層訓練病人中的驗證集，採用 `GroupShuffleSplit(test_size=0.20, random_state=777+k)`。
- [OOF 預測](results/five_fold/baseline/oof_predictions.csv.gz) 含 `encounter_id`、`patient_nbr`、`y_true`、`Fold` 和兩組預測機率，其中 `bhuvan_hgb` 為基準模型。可用這四個欄位與後續實驗逐筆對齊，再比較同一折的 AP。也可用 `--reference-oof` 與既有的 OOF 檔核對分折結果。
- [逐折指標](results/five_fold/baseline/fold_metrics.csv)、[五折摘要](results/five_fold/baseline/summary.csv)、[執行設定](results/five_fold/baseline/run.json)與[五個折別基準模型](results/five_fold/baseline/models/)均已保存。

## 重跑

從本資料夾以 Python 3.12 執行。若 CSV 已位於專案根目錄下的 `data/raw/diabetic_data.csv`，可省略 `--input`。重跑結果會寫入 `runs/` 下的新目錄。

```powershell
python -m pip install -r ../requirements.txt
python run_five_fold.py --input ../../tmp/proposal/diabetic_data.csv
```

需要重建切分時，在獨立的 Python 3.12 環境安裝 `scikit-learn==1.8.0`、`pandas==2.2.2`、`numpy==1.26.4`、`scipy==1.13.1`，執行 `python make_folds.py --input ../../tmp/proposal/diabetic_data.csv --output runs/rebuilt_folds.csv.gz`。訓練時可用 `--folds-file` 讀取重建的清單。

## 80/20 留出測試

[單次測試結果](results/holdout/test_metrics.csv) 使用 99,340 筆住院紀錄與固定的 80/20 病人互斥切分。基準模型的測試 AP 為 0.237372，原始 ICD-9 與 one-hot 編碼組為 0.248551，使用 one-hot 編碼與 15 個衍生臨床特徵的模型為 0.217896。可用 `python run.py --input ../../tmp/proposal/diabetic_data.csv` 重跑。

`upstream/` 收錄資料處理、特徵工程及切分模組與其 [MIT 授權](upstream/LICENSE)。
