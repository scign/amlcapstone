import argparse
import os
import joblib
import numpy as np
from azureml.core.run import Run
from azureml.data import DataType
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=1.0, help="Constant that multiplies the penalty terms. Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised.")
    parser.add_argument('--l1_ratio', type=float, default=1.0, help="The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.")

    args = parser.parse_args()

    data_file_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    set_column_types={
        "fixed acidity":DataType.to_float(),
        "volatile acidity":DataType.to_float(),
        "citric acid":DataType.to_float(),
        "residual sugar":DataType.to_float(),
        "chlorides":DataType.to_float(),
        "free sulfur dioxide":DataType.to_float(),
        "total sulfur dioxide":DataType.to_float(),
        "density":DataType.to_float(),
        "pH":DataType.to_float(),
        "sulphates":DataType.to_float(),
        "alcohol":DataType.to_float(),
        "quality":DataType.to_long(),
        }
    ds = TabularDatasetFactory.from_delimited_files(
        path=data_file_source,
        separator=';',
        infer_column_types=False,
        set_column_types=set_column_types
    )

    df = ds.to_pandas_dataframe().dropna()
    x = df.drop(columns='quality')
    y = df['quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
    qt = QuantileTransformer()
    qt.fit(x_train)
    x_train_scaled = qt.transform(x_train)
    x_na = ~np.isnan(x_train_scaled).any(axis=1)
    x_train_scaled = x_train_scaled[x_na]
    y_train = y_train[x_na]

    run = Run.get_context()

    run.log("Alpha", np.float(args.alpha))
    run.log("L1 Ratio", np.float(args.l1_ratio))

    y_pred = None
    try:
        model = ElasticNet(
            alpha=args.alpha,
            l1_ratio=args.l1_ratio
        ).fit(x_train_scaled, y_train)
    except ValueError as e:
        # catch incompatible parameters
        y_pred = np.zeros_like(y_test)
        run.log("Error", str(e))
    else:
        try:
            x_test_scaled = qt.transform(x_test)
            x_na = ~np.isnan(x_test_scaled).any(axis=1)
            x_test_scaled = x_test_scaled[x_na]
            y_test = y_test[x_na]
            y_pred = model.predict(x_test_scaled)
            # save the model
            os.makedirs('outputs', exist_ok=True)
            joblib.dump(model, os.path.join('outputs','model.joblib'))
            joblib.dump(qt, os.path.join('outputs','qt.joblib'))
        except Exception as e:
            run.log("Prediction Error", str(e))
    finally:
        # let's calculate all the comparable AutoML metrics
        # so that we can properly compare this model to the AutoML batch
        aml_regression_metrics = [
            'spearman_correlation',
            'normalized_root_mean_squared_error',
            'r2_score',
            'normalized_mean_absolute_error'
        ]
        run.log("spearman_correlation", spearmanr(y_test, y_pred).correlation)
        run.log("mean_squared_error", mean_squared_error(y_test, y_pred, squared=False) / (y_test.max()-y_test.min()))
        run.log("r2_score", r2_score(y_test, y_pred))

if __name__ == '__main__':
    main()