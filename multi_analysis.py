import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Load data ---
filename = 'data2.csv'  # Replace with your CSV filename
df = pd.read_csv(filename)

print("First 5 rows of data:")
print(df.head())

# --- Select numeric columns for analysis ---
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print("\nNumeric columns detected:", numeric_cols)

# --- Function to calculate manual stats ---
def manual_stats(data):
    N = len(data)
    sorted_data = sorted(data)
    
    mean = sum(sorted_data) / N
    
    if N % 2 == 0:
        median = (sorted_data[N//2 - 1] + sorted_data[N//2]) / 2
    else:
        median = sorted_data[N//2]
    
    # Mode (simple)
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
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Range": data_range,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Skewness": sk,
        "Kurtosis": kurt
    }

# --- Calculate and print stats for each numeric column ---
print("\nStatistical Measures:")
for col in numeric_cols:
    stats = manual_stats(df[col].dropna().tolist())
    print(f"\n{col}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

# --- Correlation matrix ---
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:")
print(corr)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# --- Pairplot ---
sns.pairplot(df[numeric_cols])
plt.suptitle('Pairplot of Numeric Variables', y=1.02)
plt.show()

# --- Histograms and KDE plots ---
df[numeric_cols].hist(bins=15, figsize=(15, 6), layout=(1, len(numeric_cols)), edgecolor='black')
plt.suptitle('Histograms of Numeric Variables')
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(df[col].dropna(), shade=True)
    plt.title(f'Density Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()

# --- PCA for dimensionality reduction ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols].dropna())

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'])

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', data=pc_df)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()
