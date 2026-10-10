#Imports
import pandas as pd

from .config import CONFIG

#Loading the data from the CSV file, taking file name and path from the config.py file
def load_data_from_csv():
    file_path = CONFIG.raw_data_dir / CONFIG.raw_filename
    dataframe = pd.read_csv(file_path)

    return dataframe

#Preprocessing of the data frame
def preprocess_dataframe(df, dropNa=True, dropDuplicates=True, dropColumns=None):
    #output_file_path = CONFIG.processed_data_dir / CONFIG.processed_filename
    dataframe = df.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    #Remove rows with missing values
    if dropNa:
        dataframe = dataframe.dropna()

    #Remove duplicate rows
    if dropDuplicates:
        dataframe = dataframe.drop_duplicates()

    #Remove specified columns
    if dropColumns:
        dataframe = dataframe.drop(columns=dropColumns)
        #Print the new shape of the DataFrame after dropping columns
        print("Tamaño del nuevo DataFrame:", dataframe.shape)

    #Nice to have in case later you want to avoid reprocessing the data and just load the processed data from a CSV file
    #dataframe.to_csv(output_file_path, index=False)
    print("\nDatos limpios guardados correctamente.")

    return dataframe