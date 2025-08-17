import math
import csv
from collections import Counter
import pprint

def load_dataset(filename):
    data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["CGPA"] = float(row["CGPA"])
            data.append(row)
    return data

def discretize_cgpa(cgpa):
    if cgpa >= 7.5:
        return "High"
    elif cgpa >= 6.0:
        return "Medium"
    else:
        return "Low"

def entropy(data):
    labels = [d["JobOffer"] for d in data]
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def split_info(data, attr):
    total = len(data)
    values = set(d[attr] for d in data)
    return -sum((sum(1 for d in data if d[attr] == v) / total) *
                math.log2(sum(1 for d in data if d[attr] == v) / total)
                for v in values) or 1e-10

def gain_ratio(data, attr):
    total_entropy = entropy(data)
    total = len(data)
    values = set(d[attr] for d in data)
    weighted_entropy = sum(
        (len(sub := [d for d in data if d[attr] == v]) / total) * entropy(sub)
        for v in values
    )
    gain = total_entropy - weighted_entropy
    return gain / split_info(data, attr)

def majority_class(data):
    return Counter(d["JobOffer"] for d in data).most_common(1)[0][0]

def build_tree(data, attributes):
    labels = [d["JobOffer"] for d in data]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(data)

    best_attr = max(attributes, key=lambda a: gain_ratio(data, a))
    tree = {best_attr: {}}
    for val in set(d[best_attr] for d in data):
        subset = [d for d in data if d[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(data)
        else:
            remaining = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = build_tree(subset, remaining)
    return tree

def predict(tree, sample, default):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        val = sample.get(attr)
        tree = tree[attr].get(val, default)
    return tree

if __name__ == "__main__":
    data = load_dataset("students_dataset.csv")
    for d in data:
        d["CGPA"] = discretize_cgpa(d["CGPA"])

    attributes = ["CGPA", "Interactive", "Practical", "Communication"]
    tree = build_tree(data, attributes)

    print("Decision Tree:")
    pprint.pprint(tree)

    sample = {
        "CGPA": discretize_cgpa(7.0),
        "Interactive": "Yes",
        "Practical": "Good",
        "Communication": "Moderate"
    }

    result = predict(tree, sample, default=majority_class(data))
    print(f"\nPredicted JobOffer: {result}")
