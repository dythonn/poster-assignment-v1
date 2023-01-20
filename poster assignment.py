# =============================================================================
#                               Importing Modules
# =============================================================================


import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def read_wb_data(file_path:str)->tuple:
    """
    This function reads in a file in the World Bank format and returns both the original and transposed format.
    The first element of the tuple is the original dataframe and the second element is the transposed dataframe.
    :param file_path: The file path to the World Bank data file.
    :return: A tuple containing the original dataframe and the transposed dataframe.
    """
    
    # read in the data file
    df = pd.read_csv(file_path)
    # remove duplicate rows
    df = df.drop_duplicates()
    # Remove rows with null 'Country Name'
    df = df[df['Country Name'].notna()]
    # reshape the dataframe using melt
    df = pd.melt(df, id_vars=["Country Name", "Country Code"], var_name="Year", value_name="Value")
    # remove unnecessary rows and columns
    df = df.drop(["Country Code"], axis=1)
    # Extract year from 'Year' column
    
    df['Year'] = df['Year'].str.extract(r'(\d{4})')

    df_transposed = df.T
    return df, df_transposed

# Usage
lp_data, lp_data_transposed = read_wb_data("labor participation rate.csv")
ud_data, ud_data_transposed = read_wb_data("unemployment data.csv")

print (lp_data)
print (ud_data)


# Select the "Value" column from the dataframe

data = lp_data["Value"]

data.iloc[1] = pd.to_numeric(data.iloc[1], errors='coerce')
data.dropna(inplace=True)
print (data)


data1 = ud_data["Value"]



data1.iloc[1] = pd.to_numeric(data.iloc[1], errors='coerce')
data1.dropna(inplace=True)
print (data1.T)


# Convert non-numeric values to NaN
data1 = data1.apply(pd.to_numeric, errors='coerce')

# Drop any rows that contain NaN values
data1.dropna(inplace=True)

# Drop duplicate rows
data1.drop_duplicates(inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)
data = data.values.reshape(-1,1)



# Initialize the KMeans model with the number of clusters you want
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster labels for each data point
labels = kmeans.labels_

lp_data["Cluster"] = labels

# Print the dataframe with the cluster labels
print(lp_data)

