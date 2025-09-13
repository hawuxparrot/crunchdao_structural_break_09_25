import typing
import joblib
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# extracts features from a single time series, compares properties
# before and after the boundary
def extract_features(series_data: pd.DataFrame) -> pd.Series:
    """
    Extracts statistical features from a single time series, comparing properties
    before (period=0) and after (period=1) the boundary.
    Returns np.nan for features that cannot be computed due to insufficient data.
    """
    period_0 = series_data[series_data['period'] == 0]['value']
    period_1 = series_data[series_data['period'] == 1]['value']

    features = {}
    features['period0_len'] = len(period_0)
    features['period1_len'] = len(period_1)
    features['period0_empty'] = 1 if period_0.empty else 0
    features['period1_empty'] = 1 if period_1.empty else 0
    features['period0_single_point'] = 1 if len(period_0) == 1 else 0
    features['period1_single_point'] = 1 if len(period_1) == 1 else 0


    for p, prefix in [(period_0, '0'), (period_1, '1')]:
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
        features[f'num_unique_{prefix}'] = p.nunique()

        features[f'iqr_{prefix}'] = features[f'q75_{prefix}'] - features[f'q25_{prefix}']
        features[f'range_{prefix}'] = features[f'max_{prefix}'] - features[f'min_{prefix}']
        features[f'cv_{prefix}'] = features[f'std_{prefix}'] / features[f'mean_{prefix}'] if features[f'mean_{prefix}'] != 0 else np.nan # Coefficient of Variation

    if not period_0.empty and not period_1.empty:
        features['mean_diff'] = features['mean_1'] - features['mean_0']
        features['median_diff'] = features['median_1'] - features['median_0']
        features['skew_diff'] = features['skew_1'] - features['skew_0']
        features['kurt_diff'] = features['kurt_1'] - features['kurt_0']
        features['range_diff'] = features['range_1'] - features['range_0']
        features['iqr_diff'] = features['iqr_1'] - features['iqr_0']

        # robust handling for ratios (avoid division by zero, use log for stabilization)
        features['std_ratio'] = features['std_1'] / features['std_0'] if features['std_0'] != 0 else np.nan
        features['log_std_ratio'] = np.log(features['std_ratio']) if features['std_ratio'] > 0 and np.isfinite(features['std_ratio']) else np.nan

        features['var_ratio'] = features['var_1'] / features['var_0'] if features['var_0'] != 0 else np.nan
        features['log_var_ratio'] = np.log(features['var_ratio']) if features['var_ratio'] > 0 and np.isfinite(features['var_ratio']) else np.nan

        features['iqr_ratio'] = features['iqr_1'] / features['iqr_0'] if features['iqr_0'] != 0 else np.nan
        features['log_iqr_ratio'] = np.log(features['iqr_ratio']) if features['iqr_ratio'] > 0 and np.isfinite(features['iqr_ratio']) else np.nan

        features['len_ratio'] = features['period1_len'] / features['period0_len'] if features['period0_len'] != 0 else np.nan

        features['cv_ratio'] = features['cv_1'] / features['cv_0'] if features['cv_0'] != 0 else np.nan
        features['log_cv_ratio'] = np.log(features['cv_ratio']) if features['cv_ratio'] > 0 and np.isfinite(features['cv_ratio']) else np.nan

    # Statistical Tests
    # T-test (Welch's): for mean comparison, unequal variances (robust choice)
    try:
        if len(period_0) > 1 and len(period_1) > 1: # Need at least 2 points for meaningful t-test
            _, pvalue_ttest = scipy.stats.ttest_ind(period_0.dropna(), period_1.dropna(), equal_var=False, nan_policy='omit')
            features['ttest_pvalue'] = pvalue_ttest
        else:
            features['ttest_pvalue'] = np.nan
    except ValueError:
        features['ttest_pvalue'] = np.nan

    # Levene's test: for variance comparison (robust to non-normality)
    try:
        if len(period_0) > 1 and len(period_1) > 1:
            _, pvalue_levene = scipy.stats.levene(period_0.dropna(), period_1.dropna(), center='median') # 'median' is more robust
            features['levene_pvalue'] = pvalue_levene
        else:
            features['levene_pvalue'] = np.nan
    except ValueError:
        features['levene_pvalue'] = np.nan

    # Kolmogorov-Smirnov test: for general distribution comparison
    try:
        if len(period_0) >= 1 and len(period_1) >= 1:
            _, pvalue_ks = scipy.stats.ks_2samp(period_0.dropna(), period_1.dropna())
            features['ks_pvalue'] = pvalue_ks
        else:
            features['ks_pvalue'] = np.nan
    except ValueError:
        features['ks_pvalue'] = np.nan

    # Mann-Whitney U test: non-parametric test for location (median) comparison
    try:
        if len(period_0) >= 1 and len(period_1) >= 1:
            _, pvalue_mw = scipy.stats.mannwhitneyu(period_0.dropna(), period_1.dropna(), alternative='two-sided', nan_policy='omit')
            features['mannwhitneyu_pvalue'] = pvalue_mw
        else:
            features['mannwhitneyu_pvalue'] = np.nan
    except ValueError:
        features['mannwhitneyu_pvalue'] = np.nan

    # Time Series Specific
    # Autocorrelation at Lag 1
    if features['period0_len'] > 1:
        features['autocorr_lag1_0'] = period_0.autocorr(lag=1)
    else:
        features['autocorr_lag1_0'] = np.nan
    if features['period1_len'] > 1:
        features['autocorr_lag1_1'] = period_1.autocorr(lag=1)
    else:
        features['autocorr_lag1_1'] = np.nan
    features['autocorr_lag1_diff'] = features['autocorr_lag1_1'] - features['autocorr_lag1_0']

    # Linear Regression Slope (requires at least 2 points)
    if features['period0_len'] > 1:
        slope_0, _, _, _, _ = linregress(period_0.index.get_level_values('time').values, period_0.values)
        features['slope_0'] = slope_0
    else:
        features['slope_0'] = np.nan
    if features['period1_len'] > 1:
        slope_1, _, _, _, _ = linregress(period_1.index.get_level_values('time').values, period_1.values)
        features['slope_1'] = slope_1
    else:
        features['slope_1'] = np.nan
    features['slope_diff'] = features['slope_1'] - features['slope_0']

    # Rolling Window Statistics
    rolling_windows = [5, 10, 20] # Small windows for local changes

    for window_size in rolling_windows:
        if features['period0_len'] >= window_size:
            features[f'rolling_mean_mean_{window_size}_0'] = period_0.rolling(window=window_size).mean().mean()
            features[f'rolling_std_mean_{window_size}_0'] = period_0.rolling(window=window_size).std().mean()
            features[f'rolling_max_mean_{window_size}_0'] = period_0.rolling(window=window_size).max().mean()
            features[f'rolling_min_mean_{window_size}_0'] = period_0.rolling(window=window_size).min().mean()
        else:
            features[f'rolling_mean_mean_{window_size}_0'] = np.nan
            features[f'rolling_std_mean_{window_size}_0'] = np.nan
            features[f'rolling_max_mean_{window_size}_0'] = np.nan
            features[f'rolling_min_mean_{window_size}_0'] = np.nan

        if features['period1_len'] >= window_size:
            features[f'rolling_mean_mean_{window_size}_1'] = period_1.rolling(window=window_size).mean().mean()
            features[f'rolling_std_mean_{window_size}_1'] = period_1.rolling(window=window_size).std().mean()
            features[f'rolling_max_mean_{window_size}_1'] = period_1.rolling(window=window_size).max().mean()
            features[f'rolling_min_mean_{window_size}_1'] = period_1.rolling(window=window_size).min().mean()
        else:
            features[f'rolling_mean_mean_{window_size}_1'] = np.nan
            features[f'rolling_std_mean_{window_size}_1'] = np.nan
            features[f'rolling_max_mean_{window_size}_1'] = np.nan
            features[f'rolling_min_mean_{window_size}_1'] = np.nan

        features[f'rolling_mean_mean_diff_{window_size}'] = features[f'rolling_mean_mean_{window_size}_1'] - features[f'rolling_mean_mean_{window_size}_0']
        features[f'rolling_std_mean_diff_{window_size}'] = features[f'rolling_std_mean_{window_size}_1'] - features[f'rolling_std_mean_{window_size}_0']
        features[f'rolling_max_mean_diff_{window_size}'] = features[f'rolling_max_mean_{window_size}_1'] - features[f'rolling_max_mean_{window_size}_0']
        features[f'rolling_min_mean_diff_{window_size}'] = features[f'rolling_min_mean_{window_size}_1'] - features[f'rolling_min_mean_{window_size}_0']


    return pd.Series(features)

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    # feature engineering
    X_train_features = X_train.groupby('id', group_keys=False).apply(extract_features)
    #X_train_features.fillna(X_train_features.median(), inplace=True)
    X_train_features.dropna(axis=1, how='all', inplace=True)

    imputer_values = X_train_features.median().to_dict()
    X_train_features.fillna(imputer_values, inplace=True)
    y_train_aligned = y_train.loc[X_train_features.index]

    # for our baseline t-test approach, we don't need to train a model
    # this is essentially an unsupervised approach calculated at inference time
    base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    param_dist = {
        'n_estimators': [100, 200, 300], # number of trees in the forest
        'max_depth': [10, 20, 30, None], # max depth of tree. None means nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
        'min_samples_split': [2, 5, 10], # minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4], # minimum number of samples required to be at a leaf node
        'max_features': ['sqrt', 'log2', 0.8, 1.0], # number of features to consider when looking for the best split
        'criterion': ['gini', 'entropy'] # function to measure the quality of a split
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        verbose=1,
        random_state=24,
        n_jobs=8,
        scoring='roc_auc'
    )
    random_search.fit(X_train_features, y_train_aligned)
    model = random_search.best_estimator_

    # You could enhance this by training an actual model, for example:
    # 1. Extract features from before/after segments of each time series
    # 2. Train a classifier using these features and y_train labels
    # 3. Save the trained model

    joblib.dump(model, os.path.join(model_directory_path, 'model.joblib'))
    joblib.dump(X_train_features.columns.tolist(), os.path.join(model_directory_path, 'feature_names.joblib'))
    joblib.dump(X_train_features.median().to_dict(), os.path.join(model_directory_path, 'imputer_values.joblib'))
    print("Random Forest model trained and saved.")

def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    model = joblib.load(os.path.join(model_directory_path, 'model.joblib'))
    feature_names = joblib.load(os.path.join(model_directory_path, 'feature_names.joblib'))
    imputer_values = joblib.load(os.path.join(model_directory_path, 'imputer_values.joblib'))

    yield  # Mark as ready

    # X_test can only be iterated once.
    # Before getting the next dataset, you must predict the current one.
    for dataset in X_test:
        # Baseline approach: Compute t-test between values before and after boundary point
        # The negative p-value is used as our score - smaller p-values (larger negative numbers)
        # indicate more evidence against the null hypothesis that distributions are the same,
        # suggesting a structural break
        test_features_series = extract_features(dataset)
        test_features = test_features_series.to_frame().T 
        for col in feature_names:
            if col not in test_features.columns:
                test_features[col] = np.nan
        
        """ def t_test(u: pd.DataFrame):
            return -scipy.stats.ttest_ind(
                u["value"][u["period"] == 0],  # Values before boundary point
                u["value"][u["period"] == 1],  # Values after boundary point
            ).pvalue
        """
        test_features = test_features[feature_names]
        test_features.fillna(imputer_values, inplace=True)
        prediction = model.predict_proba(test_features)[:, 1][0]
        yield prediction  # Send the prediction for the current dataset

        # Note: This baseline approach uses a t-test to compare the distributions
        # before and after the boundary point. A smaller p-value (larger negative number)
        # suggests stronger evidence that the distributions are different,
        # indicating a potential structural break.