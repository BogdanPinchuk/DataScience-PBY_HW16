import numpy as np
import apps.reporter as rpt
from pandas import Series, DataFrame
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pandas.io.formats.style import Styler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, classification_report,
                             roc_auc_score, average_precision_score)


def is_stationary(p_value: float) -> bool:
    """
    Check stationarity
    :param p_value: input value
    :return: true - stationary or false - not stationary
    """
    return p_value <= 0.05


def find_stationarity(array: np.ndarray) -> tuple[bool, int, float, np.ndarray]:
    """
    Find stationarity
    :param array: input data array
    :return: (stationarity, d - integration parameter, p-value, stationary data array)
    """
    d = 0
    stationarity = False
    output_array = array
    p_value = float("nan")

    try:
        p_value = adfuller(output_array)[1]  # type: ignore
        stationarity = is_stationary(p_value)
        # try to find stationarity
        while not stationarity:
            temp_array = np.diff(output_array)
            temp_p_value = adfuller(temp_array)[1]  # type: ignore
            d += 1
            output_array = temp_array
            p_value = temp_p_value
            stationarity = is_stationary(p_value)
    except Exception:
        pass

    return stationarity, d, p_value, output_array


def find_q_acf_manually(array: np.ndarray) -> int:
    """
    Find q manually by ACF
    :param array: input data array
    :return: q - autoregression parameter
    """
    values, limits = acf(array, nlags=30, alpha=0.05)

    q = 0
    for idx, value in enumerate(values):
        if idx == 0:
            continue

        lower, upper = limits[idx]
        # centering
        delta = upper - lower
        lower = -0.5 * delta
        upper = 0.5 * delta
        if value < lower or upper < value:
            q = idx
        else:
            break

    return q


def find_p_pacf_manually(array: np.ndarray) -> int:
    """
    Find p manually by PACF
    :param array: input data array
    :return: p - moving average parameter
    """
    values, limits = pacf(array, nlags=30, alpha=0.05)

    p = 0
    for idx, value in enumerate(values):
        if idx == 0:
            continue

        lower, upper = limits[idx]
        # centering
        delta = upper - lower
        lower = -0.5 * delta
        upper = 0.5 * delta
        if value < lower or upper < value:
            p = idx
        else:
            break

    return p


def train_test_split_by_order(array, test_size: float) -> tuple:
    """
    Split arrays or matrices into order to train and test subset
    :param array: input data
    :param test_size: the proportion of the dataset for the test split
    :return: tuple of train and test arrays
    """
    n_samples = len(array)
    if n_samples <= 1:
        raise ValueError("The array must contain more than one element!")
    n_test = int(n_samples * test_size)
    if n_test == 0:
        n_test = 1

    if isinstance(array, (Series, DataFrame)):
        train_array = array.iloc[:-n_test]
        test_array = array.iloc[-n_test:]
    else:
        train_array = array[:-n_test]
        test_array = array[-n_test:]

    return train_array, test_array


def calc_class_metrics(y_test, y_pred, y_prob=None) -> Styler:
    """
    Calc and print a classifier metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :param y_prob: probabilities set
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    rp.add_item("Confusion Matrix", rp.format_matrix(cm))
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rp.add_item("Accuracy\n(Точність)", rp.format_value(accuracy))
    # Precision
    precision = precision_score(y_test, y_pred)
    rp.add_item("Precision\n(Влучність)", rp.format_value(precision))
    # Recall
    recall = recall_score(y_test, y_pred)
    rp.add_item("Recall\n(Повнота)", rp.format_value(recall))
    # F1-score
    f1 = f1_score(y_test, y_pred)
    rp.add_item("F1-score", rp.format_value(f1))

    if y_prob is not None:
        # ROC-AUC
        auc = roc_auc_score(y_test, y_prob)
        rp.add_item("Receiver Operating Characteristic\n(ROC-AUC)", rp.format_value(auc))
        # AP
        ap = average_precision_score(y_test, y_prob)
        rp.add_item("Average Precision (AP)", rp.format_value(ap))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Метрики класифікації")
    print(classification_report(y_test, y_pred))

    return df


def calc_regres_metrics(y_test, y_pred) -> Styler:
    """
    Calc and print a regressor metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    rp.add_item("MAE", rp.format_value(mae))
    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    rp.add_item("MSE", rp.format_value(mse))
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    rp.add_item("RMSE", rp.format_value(rmse))
    # R2 - coefficient of determination
    r2 = r2_score(y_test, y_pred)
    rp.add_item("R²\n(коефіцієнт детермінації)", rp.format_value(r2))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Метрики регресії")

    return df
