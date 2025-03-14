import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Read in dataset for further processing
df = pd.read_csv("path/to/your/dataset.csv")  # Replace with your actual file path

# Extract phonological features
phonological_data = df.iloc[:, 2:]

# One-hot encode the categorical values of the phonological features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = ohe.fit_transform(phonological_data)

# Apply Truncated SVD for dimensionality reduction
n_components = 2 
svd = TruncatedSVD(n_components=n_components)
reduced_data = svd.fit_transform(encoded_data)

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(reduced_data)

# Create variable to store pairs of signs and their similarity
sign_names = df["Gloss 1"]
pairs = []
for i in range(len(sign_names)):
    for j in range(i + 1, len(sign_names)):
        pairs.append((sign_names[i], sign_names[j], similarity_matrix[i, j]))

# Sort pairs by similarity score
df_pairs = pd.DataFrame(pairs, columns=["Sign 1", "Sign 2", "Cosine Similarity"])
df_pairs = df_pairs.sort_values(by="Cosine Similarity", ascending=False)

# Save results to csv file
df_pairs.to_csv("closest_sign_pairs.csv", index=False)

# Create a t-SNE plot to visualise the embeddings
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(reduced_data)
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, label in enumerate(df['Gloss 1']):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
plt.title("Phonological Embedding Space (One-Hot + SVD)")
plt.show()