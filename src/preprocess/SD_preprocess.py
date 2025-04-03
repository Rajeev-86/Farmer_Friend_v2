import numpy as np

def handler(df):
  df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
  df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
  df['year'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())

  df.drop(columns=['Month', 'Year'], inplace=True)
  df = pd.get_dummies(df, columns=['Crop', 'Region'], drop_first=True)
  return df
