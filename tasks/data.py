from prefect import task
import pandas as pd

@task
def load_sales_data(file_path: str, date_column: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=[date_column])
    df.set_index(date_column, inplace=True)
    df.index.name = "TANGGAL"
    return df

@task
def filter_products(df: pd.DataFrame, product_id: str, target_column:str) -> pd.DataFrame:
    kode, klas, warna, ukuran = product_id.split("_")
    # Filter the DataFrame
    filter_df = df[
        (df["KODE_BARANG"] == kode) &
        (df["KLASIFIKASI_BARANG"] == klas) &
        (df["WARNA_BARANG"] == warna) &
        (df["UKURAN_BARANG"] == ukuran)
    ]

    filter_df = filter_df[[target_column]]

    return filter_df

@task
def split_data(df: pd.DataFrame, end_train: str, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets based on the end_train date.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        end_train (str): The date to split the DataFrame.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The training and testing DataFrames.
    """
    train_df = df[df.index < end_train]
    test_df = df[df.index >= end_train]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    return X_train, y_train, X_test, y_test

@task
def soft_check_sufficient_data(df: pd.DataFrame, min_data_points: int) -> bool:
    """
    Checks if the DataFrame has sufficient data points.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        min_data_points (int): The minimum number of data points required.
    
    Returns:
        bool: True if sufficient data points are present, False otherwise.
    """
    return len(df) >= min_data_points