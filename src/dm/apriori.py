import click
import pandas as pd
import numpy as np
from itertools import combinations 

# TODO: Add Pre-process
@click.command()
@click.argument("filepath")
@click.argument("field_name")
@click.argument("minsup", type=float)
def apriori(filepath, field_name, minsup):
    """
    Simple Apriori algorithm implementation.
    FILEPATH: CSV file path
    FIELD_NAME: Column containing list-like transactions
    MINSUP: Minimum support (as float between 0–1)
    """
    
    # Load dataset
    df = pd.read_csv(filepath)

    # Convert stringified lists to actual lists if needed
    data = df[field_name].apply(lambda x: eval(x) if isinstance(x, str) else x)
    data = data.tolist()
   
    #getting Unique items
    all_items = sorted(set(item for row in data for item in row))

    frequent_itemsets = []
    k = 1

    current_itemsets = [frozenset([i]) for i in all_items]

    while current_itemsets:
        itemset_counts = {}

        for transaction in data:
            tset = set(transaction)
            for itemset in current_itemsets:
                if itemset.issubset(tset):
                    itemset_counts[itemset] = itemset_counts.get(itemset, 0) + 1

        num_trans = len(data)
        frequent_k = {i: c/num_trans for i, c in itemset_counts.items() if c/num_trans >= minsup}

        if not frequent_k:
            break

        frequent_itemsets.append(frequent_k)

        current_items = list(frequent_k.keys())
        next_candidates = set()
        for i in range(len(current_items)):
            for j in range(i + 1, len(current_items)):
                union = current_items[i].union(current_items[j])
                if len(union) == k + 1:
                    next_candidates.add(union)

        current_itemsets = list(next_candidates)
        k += 1

    for level, fsets in enumerate(frequent_itemsets, start=1):
        print(f"\n== Frequent {level}-itemsets ==")
        for items, support in fsets.items():
            print(f"{set(items)}: {support:.2f}")

    # confidence

    if frequent_itemsets:
        top_level = frequent_itemsets[-1]

        best_itemset = max(top_level.items(), key=lambda x: x[1])[0]
        best_support = top_level[best_itemset]

        print(f"\nTop itemset: {set(best_itemset)} (support: {best_support:.2f})")
        print("\n== Association Rules ==")
        all_freqs = {k: v for level in frequent_itemsets for k, v in level.items()}

        # Generating confidence rules 
        for r in range(1, len(best_itemset)):
            for A in combinations(best_itemset, r):
                A = frozenset(A)
                B = best_itemset - A
                if A in all_freqs:
                    conf = best_support / all_freqs[A]
                    print(f"{set(A)} → {set(B)} (conf: {conf:.2f})")
