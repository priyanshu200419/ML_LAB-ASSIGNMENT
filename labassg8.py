import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

dataset_url = "https://www.philippe-fournier-viger.com/spmf/datasets/retail.txt"
data = []

with open(dataset_url, 'r') as file:
    for line in file:
        transaction = list(map(int, line.strip().split()))
        data.append(transaction)

item_set = set(item for transaction in data for item in transaction)
df = pd.DataFrame(columns=list(item_set), index=range(len(data)))

for i, transaction in enumerate(data):
    for item in item_set:
        df.loc[i, item] = 1 if item in transaction else 0
df.fillna(0, inplace=True)
df = df.astype(int)

min_support_values = [0.01, 0.02, 0.05, 0.1]
frequent_itemsets_results = {}

for min_support in min_support_values:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_results[min_support] = frequent_itemsets

frequent_patterns_counts = [len(frequent_itemsets_results[ms]) for ms in min_support_values]

min_confidence_values = [0.2, 0.5, 0.7, 0.9]
association_rules_counts = {}

for min_support, frequent_itemsets in frequent_itemsets_results.items():
    rules_counts = []
    for min_conf in min_confidence_values:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        rules_counts.append(len(rules))
    association_rules_counts[min_support] = rules_counts

plt.figure(figsize=(10, 5))
plt.plot(min_support_values, frequent_patterns_counts, marker='o', color='b')
plt.title('Number of Frequent Patterns vs Minimum Support')
plt.xlabel('Minimum Support')
plt.ylabel('Number of Frequent Patterns')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
for min_support, rules_counts in association_rules_counts.items():
    plt.plot(min_confidence_values, rules_counts, marker='o', label=f'Support={min_support}')
plt.title('Number of Association Rules vs Minimum Confidence')
plt.xlabel('Minimum Confidence')
plt.ylabel('Number of Association Rules')
plt.legend()
plt.grid(True)
plt.show()