# DeepHire Shield

Online recruitment fraud detection using BERT features, SMOBD-SMOTE and a neural network classifier (baseline) compared with a CNN2D extension.

## Results (test set, mean of 5 seeds)
| Model | Recall | Balanced accuracy |
|---|---|---|
| Neural Network (baseline) | 0.823 | 0.906 |
| CNN2D (extension) | 0.755 | 0.868 |

## Run the demo
    pip install -r requirements.txt
    streamlit run app.py

## Pipeline scripts
1. step1_prepare_data.py: merge, clean and split the three datasets
2. step2_extract_bert_features.py: BERT features (GPU recommended)
3. step3_split_and_smobd.py: validation split and SMOBD-SMOTE
4. step4_tune_and_evaluate.py: tuning and evaluation
5. step5_export_model.py: saves model_export/ used by app.py

Datasets are not included (Kaggle: EMSCAD, Pakistan job market, PromptCloud Indeed postings).
