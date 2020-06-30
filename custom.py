import numpy as np
import numpy_ext as npx
import pandas as pd


def load_model(code_dir):
    return "dummy"


def score(data, model, **kwargs):
    pos_label = kwargs.get('positive_class_label')
    neg_label = kwargs.get('negative_class_label')

    values = np.random.randint(1, 1000, data.shape[0])
    preds = npx.rolling_apply(np.median, min(5, data.shape[0]), values)
    preds = npx.fill_na(preds, 500)
    pred_df = pd.DataFrame()

    if pos_label and neg_label:
        pred_df[pos_label] = 1 / preds
        pred_df[neg_label] = 1 - pred_df[pos_label]
    else:
        pred_df['Predictions'] = preds

    return pred_df
