import numpy as np
import pandas as pd

df_right_foot = pd.read_csv("./data/1min_walking_test/recordV2Primary.csv")

df_left_foot = pd.read_csv("./data/1min_walking_test/recordV2Secondary.csv")

# print(df_right_foot.head())

# ensure the timestamp is numeric
df_right_foot["timestamp"] = pd.to_numeric(df_right_foot["timestamp"], errors = "coerce")

df_right_foot = df_right_foot.sort_values("timestamp")

df_left_foot["timestamp"] = pd.to_numeric(df_left_foot["timestamp"], errors = "coerce")

df_left_foot = df_left_foot.sort_values("timestamp")


# accelerometer time series plot - primary

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.plot(df_right_foot["timestamp"], df_right_foot["accel_x"], label="accel_x")

plt.plot(df_right_foot["timestamp"], df_right_foot["accel_y"], label="accel_y")

plt.plot(df_right_foot["timestamp"], df_right_foot["accel_z"], label="accel_z")

plt.title("Accelerometer signals over time - primary")
plt.xlabel("timestamp")
plt.ylabel("acceleration")
plt.legend()
plt.grid(True)

plt.show()

## accelerometer time series plot - secondary

# import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 5))

# plt.plot(df_left_foot["timestamp"], df_left_foot["accel_x"], label="accel_x")

# plt.plot(df_left_foot["timestamp"], df_left_foot["accel_y"], label="accel_y")

# plt.plot(df_left_foot["timestamp"], df_left_foot["accel_z"], label="accel_z")

# plt.title("Accelerometer signals over time - secondary")
# plt.xlabel("timestamp")
# plt.ylabel("acceleration")
# plt.legend()
# plt.grid(True)

# plt.show()

# clean the slate
# plt.figure()

## gyroscope time series plot - primary
# plt.figure(figsize=(14, 5))

# plt.plot(df_right_foot["timestamp"], df_right_foot["gyro_x"], label="gyro_x")

# plt.plot(df_right_foot["timestamp"], df_right_foot["gyro_y"], label="gyro_y")

# plt.plot(df_right_foot["timestamp"], df_right_foot["gyro_z"], label="gyro_z")

# plt.title("Gyroscope signals over time - primary")
# plt.xlabel("Timestamp")
# plt.ylabel("Anuglar velocity")

# plt.legend()
# plt.grid(True)

# plt.show()

## gyroscope time series plot - secondary
# plt.figure(figsize=(14, 5))

# plt.plot(df_left_foot["timestamp"], df_left_foot["gyro_x"], label="gyro_x")

# plt.plot(df_left_foot["timestamp"], df_left_foot["gyro_y"], label="gyro_y")

# plt.plot(df_left_foot["timestamp"], df_left_foot["gyro_z"], label="gyro_z")

# plt.title("Gyroscope signals over time - secondary")
# plt.xlabel("Timestamp")
# plt.ylabel("Anuglar velocity")

# plt.legend()
# plt.grid(True)

# plt.show()

## pressure sensors - grouped plot - primary

# import matplotlib.pyplot as plt

# pressure_cols = [f"pressure_{i:02d}" for i in range(1, 13)]

# plt.figure(figsize=(14, 6))

# for col in pressure_cols:
#     plt.plot(df_right_foot["timestamp"], df_right_foot[col], label = col)

# plt.title("Pressure sensors over time (All sensors) - primary")

# plt.xlabel("Timestamp")

# plt.ylabel("Pressure")

# plt.legend(ncol = 3, fontsize=8)

# plt.grid(True)

# plt.show()

## pressure sensors - grouped plot - secondary

# import matplotlib.pyplot as plt

# pressure_cols = [f"pressure_{i:02d}" for i in range(1, 13)]

# plt.figure(figsize=(14, 6))

# for col in pressure_cols:
#     plt.plot(df_left_foot["timestamp"], df_left_foot[col], label = col)

# plt.title("Pressure sensors over time (All sensors) - secondary")

# plt.xlabel("Timestamp")

# plt.ylabel("Pressure")

# plt.legend(ncol = 3, fontsize=8)

# plt.grid(True)

# plt.show()


## pressure sensors -> one plot per sensor (clean EDA)

# for col in pressure_cols:
#     plt.figure(figsize=(12, 3))

#     plt.plot(df_right_foot["timestamp"], df_right_foot[col])

#     plt.title(f"{col} over time")

#     plt.xlabel("Timestamp")

#     plt.ylabel("Pressure")

#     plt.grid(True)

#     plt.show


