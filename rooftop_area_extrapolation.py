import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def max_min_scale(X, mins=None, maxs=None):
    return (X-mins) / (maxs-mins)


total_inference_df = pd.read_csv("./Grid feature statistics.csv")
# region partition
# dummy_df = pd.get_dummies(total_inference_df["Partition"])
# partition_df = pd.get_dummies(total_inference_df['Partition'], prefix='Partition')  
# total_inference_df = pd.concat([total_inference_df, partition_df], axis=1)
total_inference_X = total_inference_df.drop(columns=['code', 'flag', 'Lon', 'Lat', 'Rooftop_area',
                                                     'City', 'Partition', 'Level', 'Bare_ratio',
                                                     'Dem_average',
                                                     'Dem_difference',
                                                     'Slope_average',
                                                     'Snow_ratio', 'Trees_ratio', 'Water_ratio'])

# standardization
mins = total_inference_X.min(axis=0)
maxs = total_inference_X.max(axis=0)
total_inference_X = max_min_scale(total_inference_X, mins, maxs)
X = total_inference_X[total_inference_df["flag"]==1]
target = total_inference_df[total_inference_df["flag"]==1]["Rooftop_area"].values

# modeling
rfr = RandomForestRegressor(n_estimators=100, min_samples_split=2, n_jobs=-1, random_state=1212)
rfr.fit(X, target)
total_inference_df["inference"] = rfr.predict(total_inference_X)

total_inference_df.to_csv("extrapolation_results.csv", index=None)

print(total_inference_df["inference"].sum())
