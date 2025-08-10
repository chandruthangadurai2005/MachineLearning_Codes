import math
from collections import Counter
import pprint

# Dataset
data = [
    {"CGPA": 8.1, "Interactive": "Yes", "Practical": "Very Good", "Communication": "Good", "JobOffer": "Yes"},
    {"CGPA": 6.5, "Interactive": "No", "Practical": "Avg", "Communication": "Moderate", "JobOffer": "No"},
    {"CGPA": 7.2, "Interactive": "Yes", "Practical": "Good", "Communication": "Poor", "JobOffer": "Yes"},
    {"CGPA": 5.8, "Interactive": "No", "Practical": "Avg", "Communication": "Poor", "JobOffer": "No"},
    {"CGPA": 9.0, "Interactive": "Yes", "Practical": "Very Good", "Communication": "Good", "JobOffer": "Yes"},
    {"CGPA": 6.0, "Interactive": "No", "Practical": "Good", "Communication": "Moderate", "JobOffer": "No"},
    {"CGPA": 7.8, "Interactive": "Yes", "Practical": "Very Good", "Communication": "Moderate", "JobOffer": "Yes"},
    {"CGPA": 5.5, "Interactive": "No", "Practical": "Avg", "Communication": "Poor", "JobOffer": "No"},
]

# Discretize CGPA
def discretize_cgpa(cgpa):
    if cgpa >= 7.5:
        return "High"
    elif cgpa >= 6:
        return "Medium"
    else:
        return "Low"

for record in data:
    record["CGPA"] = discretize_cgpa(record["CGPA"])

# Entropy
def entropy(data_subset):
    labels = [record["JobOffer"] for record in data_subset]
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# Information gain
def info_gain(data_subset, attribute):
    total_entropy = entropy(data_subset)
    values = set(record[attribute] for record in data_subset)
    weighted_entropy = 0.0
    total = len(data_subset)
    for val in values:
        subset = [record for record in data_subset if record[attribute] == val]
        weighted_entropy += (len(subset) / total) * entropy(subset)
    return total_entropy - weighted_entropy

# Majority class
def majority_class(data_subset):
    return Counter([record["JobOffer"] for record in data_subset]).most_common(1)[0][0]

# ID3 algorithm
def id3(data_subset, attributes):
    labels = [record["JobOffer"] for record in data_subset]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(data_subset)

    gains = [(attr, info_gain(data_subset, attr)) for attr in attributes]
    best_attr, best_gain = max(gains, key=lambda x: x[1])
    if best_gain == 0:
        return majority_class(data_subset)

    tree = {best_attr: {}}
    values = set(record[best_attr] for record in data_subset)
    for val in values:
        subset = [record for record in data_subset if record[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(data_subset)
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = id3(subset, remaining_attrs)
    return tree

# Build tree
attributes = ["CGPA", "Interactive", "Practical", "Communication"]
decision_tree = id3(data, attributes)
pprint.pprint(decision_tree)

# Prediction
def predict(tree, sample, default_class=None):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = sample.get(attribute)
    if value in tree[attribute]:
        return predict(tree[attribute][value], sample, default_class)
    else:
        return default_class

# Example prediction
new_sample = {
    "CGPA": discretize_cgpa(7.0),
    "Interactive": "Yes",
    "Practical": "Good",
    "Communication": "Moderate"
}
prediction = predict(decision_tree, new_sample, default_class=majority_class(data))
print(f"Predicted JobOffer: {prediction}")
