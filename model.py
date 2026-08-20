import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REGION_MAP = {
    'PL-Central': 'Nigeria-Central',
    'PL-North': 'Nigeria-North',
    'PL-South': 'Nigeria-South',
}

NUMERIC_FEATURES = ['price_unit', 'promotion_flag', 'delivery_days', 'stock_available', 'rolling_mean_7', 'month']
CATEGORICAL_FEATURES = ['category', 'channel', 'region', 'pack_type']
TRAIN_TEST_CUTOFF = pd.Timestamp('2024-11-01')


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].map(REGION_MAP)
    return df


def clip_negative_values(df):
    df = df.copy()
    df['units_sold'] = df['units_sold'].clip(lower=0)
    df['stock_available'] = df['stock_available'].clip(lower=0)
    df['delivered_qty'] = df['delivered_qty'].clip(lower=0)
    return df


def build_calendar_features(df):
    pieces = []
    for (sku, channel, region), g in df.groupby(['sku', 'channel', 'region']):
        g = g.set_index('date').sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        calendar = g['units_sold'].reindex(full_idx)
        shifted = calendar.shift(1)
        lag_1 = shifted
        rolling_mean_7 = shifted.rolling('7D', min_periods=3).mean()
        out = pd.DataFrame({
            'date': full_idx,
            'lag_1': lag_1.values,
            'rolling_mean_7': rolling_mean_7.values,
        })
        out['sku'] = sku
        out['channel'] = channel
        out['region'] = region
        pieces.append(out)
    calendar_features = pd.concat(pieces, ignore_index=True)
    df_fixed = df.merge(calendar_features, on=['date', 'sku', 'channel', 'region'], how='left')
    df_fixed['month'] = df_fixed['date'].dt.month
    return df_fixed


def prepare_model_frame(df_fixed):
    df_model = df_fixed.dropna(subset=['rolling_mean_7']).reset_index(drop=True)
    return df_model


def split_train_test(df_model, cutoff=TRAIN_TEST_CUTOFF):
    train = df_model[df_model['date'] < cutoff].copy()
    test = df_model[df_model['date'] >= cutoff].copy()
    return train, test


def encode_features(train, test):
    train_enc = pd.get_dummies(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
    test_enc = pd.get_dummies(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, label, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    results.append({'model': label, 'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'MAPE': round(mp, 2)})
    return mae, rmse, mp


def naive_baseline(test, results):
    subset = test.dropna(subset=['lag_1'])
    coverage = len(subset) / len(test) * 100
    label = f'Naive baseline ({coverage:.2f}% coverage)'
    evaluate(subset['units_sold'], subset['lag_1'], label, results)


def moving_average_baseline(test, results):
    evaluate(test['units_sold'], test['rolling_mean_7'], '7 day moving average', results)


def train_linear_regression(train_enc, test_enc, y_train, y_test, results):
    model = LinearRegression()
    model.fit(train_enc, y_train)
    evaluate(y_test, model.predict(test_enc), 'Linear Regression', results)
    return model


def train_random_forest(train_enc, test_enc, y_train, y_test, results):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(train_enc, y_train)
    evaluate(y_test, model.predict(test_enc), 'Random Forest (tuned)', results)
    return model


def train_gradient_boosting(train_enc, test_enc, y_train, y_test, results):
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(train_enc, y_train)
    evaluate(y_test, model.predict(test_enc), 'Gradient Boosting', results)
    return model


def feature_importance(model, train_enc, top_n=10):
    importance = pd.Series(model.feature_importances_, index=train_enc.columns).sort_values(ascending=False)
    return importance.head(top_n)


def run_pipeline(file_path):
    results = []

    df = load_data(file_path)
    df = clip_negative_values(df)
    df_fixed = build_calendar_features(df)
    df_model = prepare_model_frame(df_fixed)

    train, test = split_train_test(df_model)
    train_enc, test_enc = encode_features(train, test)
    y_train = train['units_sold']
    y_test = test['units_sold']

    naive_baseline(test, results)
    moving_average_baseline(test, results)

    train_linear_regression(train_enc, test_enc, y_train, y_test, results)
    rf_model = train_random_forest(train_enc, test_enc, y_train, y_test, results)
    train_gradient_boosting(train_enc, test_enc, y_train, y_test, results)

    results_df = pd.DataFrame(results)
    importance = feature_importance(rf_model, train_enc)

    return results_df, importance, rf_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    results_df, importance, _ = run_pipeline(args.data)

    print(results_df.to_string(index=False))
    print()
    print(importance)


if __name__ == '__main__':
    main()
