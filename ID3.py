CGPA,Interactive,Practical,Communication,JobOffer
8.5,Yes,Excellent,Good,Yes
6.2,No,Average,Moderate,No
7.3,Yes,Good,Poor,Yes
5.7,No,Average,Poor,No
9.2,Yes,Excellent,Good,Yes
6.8,No,Good,Moderate,No
7.9,Yes,Excellent,Moderate,Yes
5.4,No,Average,Poor,No
8.0,Yes,Excellent,Good,Yes
6.5,No,Average,Moderate,No
7.6,Yes,Good,Good,Yes
5.9,No,Average,Poor,No
8.7,Yes,Excellent,Good,Yes
6.1,No,Good,Moderate,No
7.4,Yes,Good,Moderate,Yes








import math
from collections import Counter
import pprint
import csv

# Load dataset from CSV
def load_dataset(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["CGPA"] = float(row["CGPA"])
            data.append(row)
    return data

# Discretize CGPA
def discretize_cgpa(cgpa):
    if cgpa >= 7.5:
        return "High"
    elif cgpa >= 6:
        return "Medium"
    else:
        return "Low"

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

if __name__ == "__main__":
    # Load and preprocess dataset
    data = load_dataset("students_dataset.csv")
    for record in data:
        record["CGPA"] = discretize_cgpa(record["CGPA"])

    # Build decision tree
    attributes = ["CGPA", "Interactive", "Practical", "Communication"]
    decision_tree = id3(data, attributes)

    print("Decision Tree:")
    pprint.pprint(decision_tree)

    # Test prediction
    new_sample = {
        "CGPA": discretize_cgpa(7.0),
        "Interactive": "Yes",
        "Practical": "Good",
        "Communication": "Moderate"
    }
    prediction = predict(decision_tree, new_sample, default_class=majority_class(data))
    print(f"\nPredicted JobOffer for {new_sample}: {prediction}")
