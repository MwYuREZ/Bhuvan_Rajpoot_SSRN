# Data provenance

Source: **Diabetes 130-US Hospitals for Years 1999-2008**, UCI dataset 296.

- Official landing page: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
- Download URL: https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
- Dataset citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Required `diabetic_data.csv` SHA-256: `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`
- Original CSV: 101,766 encounters, 50 columns including outcome and identifiers.

Run `python scripts/download_data.py` to download and verify the files in `data/raw/`. The main runner also downloads them if needed. Raw CSV files are excluded from Git.

Only `?` is parsed as missing. The literal string `None` in `A1Cresult` and `max_glu_serum` means a test was not measured and is retained as a category. Default pandas missing-value parsing would erase that distinction.

The cohort removes invalid gender and discharge dispositions 11, 13, 14, 19, 20, 21. The audit records both exclusion flags independently; exclusions can overlap and should not be summed without accounting for overlap. The source row order is retained. Patient IDs are used for grouping and exported for audit, never as predictors.
