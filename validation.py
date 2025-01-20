# from itertools import combinations
# from collections import defaultdict

# def buc_top_down(data, dimensions, depth_limit=3, depth=0):
#     """
#     Implements the BUC algorithm for cuboid enumeration.
    
#     Args:
#         data (list of dict): The dataset with multidimensional data.
#         dimensions (list): List of dimensions to explore.
#         depth_limit (int): Maximum depth of the lattice.
#         depth (int): Current depth in the lattice.

#     Returns:
#         list of dict: Enumerated cuboids.
#     """
#     if depth > depth_limit or not dimensions:
#         return []

#     # Group data by the current set of dimensions
#     grouped_data = defaultdict(list)
#     for record in data:
#         key = tuple(record[dim] for dim in dimensions)
#         grouped_data[key].append(record)

#     # Current cuboid
#     current_cuboid = list(grouped_data.keys())

#     # Recursive exploration for next level cuboids
#     sub_cuboids = []
#     for i in range(len(dimensions)):
#         # Remove one dimension at a time
#         remaining_dims = dimensions[:i] + dimensions[i + 1 :]
#         sub_cuboids.extend(buc_top_down(data, remaining_dims, depth_limit, depth + 1))

#     # Combine current and sub-cuboids
#     return [current_cuboid] + sub_cuboids


# # Example Usage:
# if __name__ == "__main__":
#     # Example dataset (each record is a dictionary)
#     data = [
#         {"A": "a1", "B": "b1", "C": "c1"},
#         {"A": "a1", "B": "b1", "C": "c2"},
#         {"A": "a2", "B": "b2", "C": "c1"},
#         {"A": "a2", "B": "b1", "C": "c2"},
#     ]
#     dimensions = ["A", "B", "C"]
#     depth_limit = 3

#     # Run the BUC algorithm
#     cuboids = buc_top_down(data, dimensions, depth_limit)
#     print("Enumerated Cuboids:", cuboids)
import pandas as pd
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt

# Corrected Data
data = {
    'group': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'values': [10, 12, 11, 20, 21, 19, 30, 31, 29]
}
df = pd.DataFrame(data)

# Group values by group name
groups = df.groupby('group')['values']

# Perform Kruskal-Wallis test
kruskal_results = kruskal(*(groups.get_group(g) for g in groups.groups))

# Visualize data using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='group', y='values', palette='coolwarm')
plt.title('Value Distribution Across Groups')
plt.xlabel('Group')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate group labels for clarity
plt.show()

# Print Kruskal-Wallis Test Results
print(kruskal_results)
