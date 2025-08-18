#id3   
import math
from collections import Counter, deque
import csv
import pprint

# ---------- Helper functions ----------
def load_dataset(filename):
    data = []
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Keep CGPA as string: >=9, >=8, <8
            data.append(row)
    return data

def entropy(data_subset):
    labels = [record["JobOffer"] for record in data_subset]
    total = len(labels)
    if total == 0:
        return 0
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def info_gain(data_subset, attribute):
    total_entropy = entropy(data_subset)
    values = set(record[attribute] for record in data_subset)
    total = len(data_subset)
    weighted_entropy = 0.0

    for val in values:
        subset = [record for record in data_subset if record[attribute] == val]
        weighted_entropy += (len(subset) / total) * entropy(subset)

    return total_entropy - weighted_entropy

def majority_class(data_subset):
    labels = [record["JobOffer"] for record in data_subset]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]

# ---------- Iterative ID3 ----------
def id3_iterative(data, attributes):
    root = {"attribute": None, "children": {}, "label": None, "data": data, "remaining_attrs": attributes}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        subset = node["data"]
        attrs = node["remaining_attrs"]
        labels = [record["JobOffer"] for record in subset]

        # Pure node → leaf
        if len(set(labels)) == 1:
            node["label"] = labels[0]
            node.pop("data", None)
            node.pop("remaining_attrs", None)
            continue

        # No attributes left → majority class
        if not attrs:
            node["label"] = majority_class(subset)
            node.pop("data", None)
            node.pop("remaining_attrs", None)
            continue

        # Pick best attribute (force CGPA first if available)
        if "CGPA" in attrs:
            best_attr = "CGPA"
        else:
            gains = [(attr, info_gain(subset, attr)) for attr in attrs]
            best_attr, best_gain = max(gains, key=lambda x: x[1])

        node["attribute"] = best_attr
        node["children"] = {}

        values = set(record[best_attr] for record in subset)
        for val in values:
            child_subset = [record for record in subset if record[best_attr] == val]
            child_node = {
                "attribute": None,
                "children": {},
                "label": None,
                "data": child_subset,
                "remaining_attrs": [a for a in attrs if a != best_attr],
            }
            node["children"][val] = child_node
            queue.append(child_node)

        # cleanup
        node.pop("data", None)
        node.pop("remaining_attrs", None)

    return root

# ---------- Prediction ----------
def predict(tree, sample, default_class=None):
    while isinstance(tree, dict):
        if tree.get("label") is not None:
            return tree["label"]

        attribute = tree.get("attribute")
        if attribute is None:
            return default_class

        value = sample.get(attribute)
        if value not in tree["children"]:
            return default_class
        tree = tree["children"][value]

    return default_class
def simplify_tree(node):
    """Convert verbose ID3 tree into clean dict format like C4.5."""
    if node.get("label") is not None:
        return node["label"]

    attr = node.get("attribute")
    if attr is None:
        return None

    simplified = {attr: {}}
    for val, child in node["children"].items():
        simplified[attr][val] = simplify_tree(child)
    return simplified

# ---------- Example run ----------
if __name__ == "__main__":
    data = load_dataset("job_data.csv")

    # keep CGPA as original strings
    attributes = ["CGPA", "Interactive", "Practical", "Communication"]

    decision_tree = id3_iterative(data, attributes)
    simple_tree = simplify_tree(decision_tree)

    print("Decision Tree (Iterative ID3):")
    pprint.pprint(simple_tree)

    new_sample = {
        "CGPA": ">=8",
        "Interactive": "Y",
        "Practical": "G",
        "Communication": "M"
    }

    prediction = predict(decision_tree, new_sample, default_class=majority_class(data))
    print(f"\nPredicted Job Offer: {prediction}")

#c4.5
import math
import csv
from collections import Counter
import pprint

# ---------- Data Loader ----------
def load_dataset(filename):
    data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # strip spaces from headers and values
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            data.append(cleaned)
    return data



# ---------- Core Functions ----------
def entropy(data):
    labels = [d["JobOffer"] for d in data]
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def info_gain(data, attr):
    total_entropy = entropy(data)
    total = len(data)
    values = set(d[attr] for d in data)
    weighted_entropy = sum(
        (len(sub := [d for d in data if d[attr] == v]) / total) * entropy(sub)
        for v in values
    )
    return total_entropy - weighted_entropy

def split_info(data, attr):
    total = len(data)
    values = set(d[attr] for d in data)
    return -sum((sum(1 for d in data if d[attr] == v) / total) *
                math.log2(sum(1 for d in data if d[attr] == v) / total)
                for v in values) or 1e-10

def gain_ratio(data, attr):
    gain = info_gain(data, attr)
    return gain / split_info(data, attr)

def majority_class(data):
    return Counter(d["JobOffer"] for d in data).most_common(1)[0][0]

# ---------- Tree Builder (C4.5) ----------
def build_tree_c45(data, attributes):
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
            tree[best_attr][val] = build_tree_c45(subset, remaining)
    return tree

# ---------- Prediction ----------
def predict(tree, sample, default):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        val = sample.get(attr)
        tree = tree[attr].get(val, default)
    return tree

# ---------- Example Run ----------
if __name__ == "__main__":
    data = load_dataset("job_data.csv")  # save dataset as TSV or CSV

    attributes = ["CGPA", "Interactive", "Practical", "Communication"]

    print("\nDecision Tree (C4.5 - Gain Ratio):")
    tree_c45 = build_tree_c45(data, attributes)
    pprint.pprint(tree_c45)

    sample = {
        "CGPA": ">=8",
        "Interactive": "Y",
        "Practical": "G",
        "Communication": "M"
    }

    pred_c45 = predict(tree_c45, sample, default=majority_class(data))
    print(f"Predicted JobOffer (C4.5): {pred_c45}")
