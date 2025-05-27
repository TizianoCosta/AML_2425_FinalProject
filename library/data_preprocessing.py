# Useful package
import requests
import zipfile
import io
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# Code for preprocessing of the dataset and for the pipeline

def load_training_data(x_path, y_path):
  """
  Loads X_train.csv and the second column of y_train.csv into a single pandas DataFrame.

  Args:
    x_path (str): The path to the X_train.csv file.
    y_path (str): The path to the y_train.csv file.

  Returns:
    pandas.DataFrame: A DataFrame containing the data from X_train.csv
                      and the second column of y_train.csv.
  """
  x_train = pd.read_csv(x_path)
  y_train = pd.read_csv(y_path)

  # Assuming y_train has at least 2 columns and the second column is at index 1
  if y_train.shape[1] > 1:
    combined_data = x_train.copy()
    combined_data['y_target'] = y_train.iloc[:, 1]
    return combined_data
  else:
    print("Error: y_train.csv does not have a second column.")
    return x_train

# x_train_path = 'dataset/X_train.csv'
# y_train_path = 'dataset/y_train.csv'

# data_train = load_training_data(x_train_path, y_train_path)



def rename_dataframe_columns(df, new_column_names):
  """
  Renames the columns of a pandas DataFrame.

  Args:
    df: The pandas DataFrame whose columns are to be renamed.
    new_column_names: A list of new column names. The length of this list
                      must match the number of columns in the DataFrame.

  Returns:
    The DataFrame with renamed columns.
  """
  if len(new_column_names) != len(df.columns):
    raise ValueError("The number of new column names must match the number of existing columns.")
  df.columns = new_column_names
  return df

# new_names_for_train_data = ['idx', 'torque_meas', 'outside_air_temp', 'mean_gas_temp', 'power_avail', 'indicated_air_speed', 'net_power', 'compressor_speed', 'health_state'] # Replace with your desired names

# data_train = rename_dataframe_columns(data_train, new_names_for_train_data)

# Aggiungere il drop della colonna di indice
# data_train = data_train.drop('idx', axis=1)


#######################################################################
#######################################################################
#######################################################################

# Da qui partono funzioni relative a trasformazioni di singole feature


def transform_with_custom_root(df, column_name, root_degree):
  """
  Applies a custom root transformation (1/root_degree power) to a column.
  Handles positive, negative, and zero values appropriately based on the root degree.

  Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to transform.
    root_degree (float): The degree of the root (e.g., 2 for square root, 3 for cube root).

  Returns:
    pd.DataFrame: The DataFrame with the transformed column.
  """
  new_column_name = f'{column_name}_root_{root_degree:.2f}_transformed'

  if root_degree == 0:
      raise ValueError("Root degree cannot be zero.")
  elif root_degree % 2 == 0:  # Even root
      # For even roots, we can only take the root of non-negative numbers
      if (df[column_name] < 0).any():
          print(f"Warning: Column '{column_name}' contains negative values. Cannot apply even root directly.")
          # You might choose to handle this by taking the root of the absolute value,
          # or setting negative values to NaN, depending on your data context.
          # Here, we'll take the root of the absolute value for demonstration.
          df[new_column_name] = np.power(np.abs(df[column_name]), 1/root_degree)
      else:
          df[new_column_name] = np.power(df[column_name], 1/root_degree)
  else:  # Odd root
      # Odd roots can handle positive, negative, and zero values
      df[new_column_name] = np.sign(df[column_name]) * np.power(np.abs(df[column_name]), 1/root_degree)

  return df

# Example usage with a custom root (e.g., 1.5)
# custom_root_degree = 2.35
# data_train = transform_with_custom_root(data_train.copy(), 'power_avail', custom_root_degree)


def create_binned_qualitative_variable(df, column_name, num_bins, strategy='quantile'):
  """
  Creates a qualitative (categorical) variable by binning a numerical column.

  Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the numerical column to bin.
    num_bins (int): The desired number of bins.
    strategy (str): The strategy to use for binning. 'quantile' uses quantiles
                    to ensure bins have approximately equal numbers of observations.
                    'uniform' creates bins with equal widths. Default is 'quantile'.

  Returns:
    pd.DataFrame: The DataFrame with a new qualitative column.
                  The new column name will be f'{column_name}_binned_{num_bins}_{strategy}'.
  """
  if column_name not in df.columns:
    raise ValueError(f"La colonna '{column_name}' non è presente nel DataFrame.")
  if num_bins <= 1:
      raise ValueError("Il numero di bins deve essere maggiore di 1.")

  new_column_name = f'{column_name}_binned_{num_bins}_{strategy}'

  if strategy == 'quantile':
    # Use qcut to create bins based on quantiles (approximately equal number of observations)
    # `duplicates='drop'` handles cases where quantile boundaries are not unique,
    # which can happen with skewed or discrete data.
    df[new_column_name] = pd.qcut(df[column_name], q=num_bins, labels=False, duplicates='drop')
  elif strategy == 'uniform':
    # Use cut to create bins of equal width
    df[new_column_name] = pd.cut(df[column_name], bins=num_bins, labels=False, include_lowest=True)
  else:
    raise ValueError(f"Strategia di binning non valida: '{strategy}'. Scegliere tra 'quantile' o 'uniform'.")

  # Convert the binned column to object/category type if needed, or keep as int for simplicity
  # Here we keep it as int representing the bin number

  return df

# Example usage for 'indicated_air_speed':
# num_bins_indicated_air_speed = 5 # Define the number of bins
# binning_strategy = 'quantile' # Or 'uniform'

#data_train = create_binned_qualitative_variable(
#    data_train.copy(),
#    'indicated_air_speed',
#    num_bins_indicated_air_speed,
#    strategy=binning_strategy
#)




## PCA per indicated_air_speed e compressor_speed
# Select the columns for PCA
features_for_pca = data_train[['compressor_speed', 'net_power']]
# Initialize PCA with 1 component (to combine the two variables)
pca = PCA(n_components=1)
# Fit PCA on the selected features and transform them
data_train['compressor_speed_net_power_pca'] = pca.fit_transform(features_for_pca)



## Creazione di torque_times_temp

data_train['torque_times_temp'] = data_train['torque_meas'] * data_train['outside_air_temp']




# Standardizzazione
def standardize_columns(df, columns_to_standardize):
  """
  Standardizes specified columns of a pandas DataFrame to have values between 0 and 1
  using MinMaxScaler.

  Args:
    df: The pandas DataFrame to standardize.
    columns_to_standardize: A list of column names to standardize.

  Returns:
    The DataFrame with the specified columns standardized.
  """
  scaler = MinMaxScaler()
  df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
  return df

# Example usage:
# Assuming you want to standardize all numerical columns except the index and the target variable
# Identify numerical columns (excluding 'idx' and 'health_state' in this case)
# numerical_cols = data_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
# columns_to_standardize = [col for col in numerical_cols if col not in ['idx', 'health_state']]





# Creazione pipeline
def prepare_data_pipeline(x_path, y_path, new_column_names=None,
                          root_transformations=None,
                          binning_config=None,
                          standardize=True,
                          drop_index_col='idx'):
    """
    Esegue la pipeline completa di preprocessing.
    
    Args:
        x_path (str): path al file X_train.csv
        y_path (str): path al file y_train.csv
        new_column_names (list): lista di nuovi nomi colonne (opzionale)
        root_transformations (dict): dict {colonna: radice}
        binning_config (dict): dict {colonna: (num_bins, strategia)}
        standardize (bool): se standardizzare le colonne numeriche
        drop_index_col (str): nome della colonna da droppare (opzionale)
        
    Returns:
        pd.DataFrame: DataFrame preprocessato pronto per il training
    """
    df = load_training_data(x_path, y_path)

    if new_column_names:
        df = rename_dataframe_columns(df, new_column_names + ['y_target'])

    if drop_index_col in df.columns:
        df = df.drop(drop_index_col, axis=1)

    # Trasformazioni custom root
    if root_transformations:
        for col, deg in root_transformations.items():
            df = transform_with_custom_root(df, col, deg)

    # Binning
    if binning_config:
        for col, (n_bins, strategy) in binning_config.items():
            df = create_binned_qualitative_variable(df, col, n_bins, strategy)

    # PCA: esempio hardcoded ma puoi parametrizzare se vuoi
    if {'compressor_speed', 'net_power'}.issubset(df.columns):
        pca = PCA(n_components=1)
        df['compressor_speed_net_power_pca'] = pca.fit_transform(df[['compressor_speed', 'net_power']])

    # Feature engineering manuale
    if {'torque_meas', 'outside_air_temp'}.issubset(df.columns):
        df['torque_times_temp'] = df['torque_meas'] * df['outside_air_temp']

    # Standardizzazione
    if standardize:
        target = 'y_target'
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        columns_to_standardize = [col for col in numerical_cols if col != target]
        df = standardize_columns(df, columns_to_standardize)

    return df



### Esempio di utilizzo della pipeline
x_path = 'dataset/X_train.csv'
y_path = 'dataset/y_train.csv'

# Configurazioni opzionali
new_column_names = ['idx', 'torque_meas', 'outside_air_temp', 'mean_gas_temp',
                    'power_avail', 'indicated_air_speed', 'net_power', 'compressor_speed']

root_transform = {'power_avail': 2.35}
binning = {'indicated_air_speed': (5, 'quantile')}

data_ready = prepare_data_pipeline(
    x_path, y_path,
    new_column_names=new_column_names,
    root_transformations=root_transform,
    binning_config=binning
)

# data_ready ora è pronto per essere usato in un modello