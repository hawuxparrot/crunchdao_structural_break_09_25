import os
import typing
import joblib
import pandas as pd
import numpy as np
import scipy.stats
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import warnings
from scipy.stats import linregress

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def extract_features(series_data: pd.DataFrame) -> pd.Series:
    period_0 = series_data[series_data['period'] == 0]['value'].dropna()
    period_1 = series_data[series_data['period'] == 1]['value'].dropna()
    features = {}
    for p, prefix in [(period_0, '0'), (period_1, '1')]:
        if not p.empty:
            features[f'mean_{prefix}'] = p.mean()
            features[f'std_{prefix}'] = p.std()
            features[f'var_{prefix}'] = p.var()
            features[f'skew_{prefix}'] = p.skew()
            features[f'kurt_{prefix}'] = p.kurt()
            features[f'median_{prefix}'] = p.median()
            features[f'q25_{prefix}'] = p.quantile(0.25)
            features[f'q75_{prefix}'] = p.quantile(0.75)
            features[f'min_{prefix}'] = p.min()
            features[f'max_{prefix}'] = p.max()
            features[f'abs_mean_{prefix}'] = p.abs().mean()
            features[f'abs_std_{prefix}'] = p.abs().std()
            features[f'abs_max_{prefix}'] = p.abs().max()
            features[f'autocorr_lag1_{prefix}'] = p.autocorr(lag=1)
            features[f'autocorr_lag5_{prefix}'] = p.autocorr(lag=5)
        else:
            # Fill with zeros or NaNs if a period is empty
            keys = ['mean', 'std', 'var', 'skew', 'kurt', 'median', 'q25', 'q75', 'min', 'max', 'abs_mean', 'abs_std', 'abs_max', 'autocorr_lag1', 'autocorr_lag5']
            for key in keys:
                features[f'{key}_{prefix}'] = np.nan

    if not period_0.empty and not period_1.empty:
        features['mean_diff'] = features['mean_1'] - features['mean_0']
        features['median_diff'] = features['median_1'] - features['median_0']
        features['std_ratio'] = features['std_1'] / features['std_0'] if features['std_0'] > 0 else np.nan
        features['var_ratio'] = features['var_1'] / features['var_0'] if features['var_0'] > 0 else np.nan
        features['abs_mean_ratio'] = features['abs_mean_1'] / features['abs_mean_0'] if features['abs_mean_0'] > 0 else np.nan
        features['autocorr_lag1_diff'] = features['autocorr_lag1_1'] - features['autocorr_lag1_0']

        # T-test for mean comparison
        ttest_stat, ttest_pvalue = scipy.stats.ttest_ind(period_0, period_1, equal_var=False)
        features['ttest_pvalue'] = ttest_pvalue
        features['ttest_stat'] = ttest_stat

        # Levene's test for variance comparison
        levene_stat, levene_pvalue = scipy.stats.levene(period_0, period_1)
        features['levene_pvalue'] = levene_pvalue
        features['levene_stat'] = levene_stat

    return pd.Series(features)

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
): 
    X_train_features = X_train.groupby('id', group_keys=False).apply(extract_features)
    imputer_values = X_train_features.median().to_dict()
    X_train_features.fillna(imputer_values, inplace=True)
    y_train_aligned = y_train.loc[X_train_features.index]

    base_model = lgb.LGBMClassifier(random_state=24, class_weight='balanced')

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 30, 40, 50],
        'max_depth': [-1, 10, 20],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator = base_model,
        param_distributions = param_dist,
        n_iter = 50,
        cv = 5,
        verbose = 1,
        random_state = 24,
        n_jobs = -1,
        scoring = 'roc_auc'
    )
    random_search.fit(X_train_features, y_train_aligned)
    model = random_search.best_estimator_
    joblib.dump(model, os.path.join(model_directory_path, 'model.joblib'))
    joblib.dump(X_train_features.columns.tolist(), os.path.join(model_directory_path, 'feature_names.joblib'))
    joblib.dump(imputer_values, os.path.join(model_directory_path, 'imputer_values.joblib'))
    print("LightGBM model trained and saved.")

def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    model = joblib.load(os.path.join(model_directory_path, 'model.joblib'))
    feature_names = joblib.load(os.path.join(model_directory_path, 'feature_names.joblib'))
    imputer_values = joblib.load(os.path.join(model_directory_path, 'imputer_values.joblib'))

    yield  # Mark as ready

    for dataset in X_test:
        test_features_series = extract_features(dataset)
        test_features = test_features_series.to_frame().T
        
        # Align columns with the training data
        for col in feature_names:
            if col not in test_features.columns:
                test_features[col] = imputer_values.get(col, 0) # Use imputer value for missing columns

        test_features = test_features[feature_names]
        test_features.fillna(imputer_values, inplace=True)
        
        prediction = model.predict_proba(test_features)[:, 1][0]
        yield prediction