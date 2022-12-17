import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pork_data = pd.read_csv("./Pork_retail_price.csv")
temp_data = pd.read_csv("./temperature.csv")

pork = np.array(pork_data)

temp = np.array(temp_data)

data = np.empty(shape=(0,3), dtype=np.object_)

for t_date in temp:
    for p_date in pork:
        if t_date[0] == p_date[0]:
            final_data = np.insert(t_date, len(t_date), p_date[1], axis=0)
            data = np.insert(data, len(data), [final_data], axis=0)
            
# print(data[0])
# print(data[1])
# print(data[2])
# print(data)

df = pd.DataFrame(data)
df.to_csv('sample.csv', index=False)