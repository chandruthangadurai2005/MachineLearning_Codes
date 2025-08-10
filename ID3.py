import math
from collections import Counter

# Sample dataset (slightly changed values)
dataset = [
    {"CGPA": 8.3, "Interactive": "Yes", "Practical": "Excellent", "Communication": "Good", "JobOffer": "Yes"},
    {"CGPA": 6.4, "Interactive": "No", "Practical": "Average", "Communication": "Moderate", "JobOffer": "No"},
    {"CGPA": 7.0, "Interactive": "Yes", "Practical": "Good", "Communication": "Poor", "JobOffer": "Yes"},
    {"CGPA": 5.9, "Interactive": "No", "Practical": "Average", "Communication": "Poor", "JobOffer": "No"},
    {"CGPA": 9.1, "Interactive": "Yes", "Practical": "Excellent", "Communication": "Good", "JobOffer": "Yes"},
    {"CGPA": 6.1, "Interactive": "No", "Practical": "Good", "Communication": "Moderate", "JobOffer": "No"},
    {"CGPA": 7.9, "Interactive": "Yes", "Practical": "Excellent", "Communication": "Moderate", "JobOffer": "Yes"},
    {"CGPA": 5.6, "Interactive": "No", "Practical": "Average", "Communication": "Poor", "JobOffer": "No"},
]

# Function to bucket CGPA into categories
def categorize_cgpa(score):
    if score >= 7.5:
        return "High"
    elif score >= 6.0:
        return "Medium"
    else:
        return "Low"

# Apply categorization to all records
for row in dataset:
    row["CGPA"] = categorize_cgpa(row["CGPA"])

# Function to calculate entropy
def calc_entropy(records):
    labels = [row["JobOffer"] for row in records]
    total = len(labels)
    counts = Counter(labels)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# Function to calculate information gain for a given attribute
def calc_info_gain(records, attribute):
    base_entropy = calc_entropy(records)
    values = set(row[attribute] for row in records)
    weighted_entropy = 0
    total = len(records)
    for val in values:
        subset = [row for row in records if row[attribute] == val]
        weighted_entropy += (len(subset) / total) * calc_entropy(subset)
    return base_entropy - weighted_entropy

# Function to find majority class
def most_common_label(records):
    return Counter([row["JobOffer"] for row in records]).most_common(1)[0][0]

# Recursive ID3 function
def build_tree(records, attributes):
    labels = [row["JobOffer"] for row in records]
    # If all labels are same → return that label
    if len(set(labels)) == 1:
        return labels[0]
    # If no attributes left → return majority label
    if not attributes:
        return most_common_label(records)
    
    # Select attribute with highest info gain
    gains = [(attr, calc_info_gain(records, attr)) for attr in attributes]
    best_attr, best_gain = max(gains, key=lambda x: x[1])
    if best_gain == 0:
        return most_common_label(records)
    
    tree = {best_attr: {}}
    for val in set(row[best_attr] for row in records):
        subset = [row for row in records if row[best_attr] == val]
        if not subset:
            tree[best_attr][val] = most_common_label(records)
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = build_tree(subset, remaining_attrs)
    return tree

# Attributes to use for splitting
features = ["CGPA", "Interactive", "Practical", "Communication"]

# Build and print the decision tree
decision_tree = build_tree(dataset, features)

import pprint
pprint.pprint(decision_tree)
