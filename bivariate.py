import csv
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Read data from CSV file
temperatures = []
humidities = []

with open('data1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        if row[0] == '' or row[1] == '':
            continue
        temperatures.append(float(row[0]))
        humidities.append(float(row[1]))

# Function to manually compute statistics
def manual_stats(data):
    N = len(data)
    sorted_data = sorted(data)
    
    mean = sum(sorted_data) / N
    
    if N % 2 == 0:
        median = (sorted_data[N//2 - 1] + sorted_data[N//2]) / 2
    else:
        median = sorted_data[N//2]
    
    counts = {}
    for val in sorted_data:
        counts[val] = counts.get(val, 0) + 1
    max_count = max(counts.values())
    mode = [k for k, v in counts.items() if v == max_count]
    if max_count == 1:
        mode = "No mode"
    
    data_range = max(sorted_data) - min(sorted_data)
    
    variance = sum((x - mean) ** 2 for x in sorted_data) / (N - 1)
    std_dev = variance ** 0.5
    
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    iqr = q3 - q1
    
    sk = skew(sorted_data)
    kurt = kurtosis(sorted_data)
    
    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "range": data_range,
        "variance": variance,
        "std_dev": std_dev,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": sk,
        "kurtosis": kurt
    }

# Print stats in a clean format
def print_stats(label, stats):
    print(f"{label} Stats:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")
    print()  # blank line for spacing

temps_stats = manual_stats(temperatures)
humid_stats = manual_stats(humidities)

print_stats("Temperatures", temps_stats)
print_stats("Humidities", humid_stats)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
print(sns.__version__)
# Convert data to DataFrame for seaborn
data = pd.DataFrame({
    'Temperature': temperatures,
    'Humidity': humidities
})

# Apply seaborn style
sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))

# Scatterplot with regression line (lmplot needs separate call)
sns.lmplot(x='Temperature', y='Humidity', data=data, aspect=1.5, ci=None, scatter_kws={"s": 60, "alpha": 0.7})
plt.title('Temperature vs Humidity with Regression Line')
plt.show()

# Histogram with KDE for Temperature
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['Temperature'], kde=True, color='orange', bins=5)
plt.title('Temperature Distribution')

# Histogram with KDE for Humidity
plt.subplot(1,2,2)
sns.histplot(data['Humidity'], kde=True, color='green', bins=5)
plt.title('Humidity Distribution')

plt.tight_layout()
plt.show()
