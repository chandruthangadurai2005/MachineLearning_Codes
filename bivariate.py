import csv
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

temperatures = []
humidities = []

with open('data1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        if len(row) < 2:
            continue
        if row[0] == '' or row[1] == '':
            continue
        temperatures.append(float(row[0]))
        humidities.append(float(row[1]))

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

def print_stats(label, stats):
    print(f"{label} Stats:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")
    print()

temps_stats = manual_stats(temperatures)
humid_stats = manual_stats(humidities)

print_stats("Temperatures", temps_stats)
print_stats("Humidities", humid_stats)

data = pd.DataFrame({
    'Temperature': temperatures,
    'Humidity': humidities
})

sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))
sns.lmplot(x='Temperature', y='Humidity', data=data, aspect=1.5, ci=None, scatter_kws={"s": 60, "alpha": 0.7})
plt.title('Temperature vs Humidity with Regression Line')
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['Temperature'], kde=True, color='orange', bins=5)
plt.title('Temperature Distribution')
plt.subplot(1,2,2)
sns.histplot(data['Humidity'], kde=True, color='green', bins=5)
plt.title('Humidity Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
