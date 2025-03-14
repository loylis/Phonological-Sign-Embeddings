import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Read in dataset for further processing
df = pd.read_csv("path/to/your/dataset.csv")

# Extract phonological features
phonological_features = df.iloc[:, 2:].astype(str)  # Adjust columns to match your dataset

# Save phonological features to a list of lists
sequences = phonological_features.apply(lambda row: list(row), axis=1).tolist()

# Check whether sequences are unique
unique_features = set(item for seq in sequences for item in seq)
unique_sequences = set(tuple(seq) for seq in sequences)
print(f"Unique phonological features: {len(unique_features)}")
print(f"Unique phonological sequences: {len(unique_sequences)}")

# Create a dictionary that maps each feature sequence to its gloss
sequence_to_gloss = {" ".join(seq): gloss for seq, gloss in zip(sequences, df["Gloss 1"].tolist())}

# Set up and train Word2Vec model
word_sequences = [[" ".join(row)] for row in sequences]
embedding_dim = 50 # Adjust according to desired embedding size
w2v_model = Word2Vec(sentences=word_sequences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Save the Word2vec model
w2v_model.save("phonological_embeddings.model")

# Extract gloss labels and create a mapping from word to gloss
labels = df["Gloss 1"].tolist()
word_to_gloss = {word: sequence_to_gloss[word] for word in words if word in sequence_to_gloss}

# Get embeddings for glosses
gloss_embeddings = np.array([w2v_model.wv[word] for word in words if word in word_to_gloss])

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(gloss_embeddings)

# Find closest gloss for each gloss
closest_glosses = []
for i, label in enumerate(filtered_labels):
    similarities = similarity_matrix[i]
    sorted_indices = np.argsort(similarities)[::-1]
    closest_index = next((idx for idx in sorted_indices if idx != i), None)

    if closest_index is not None:
        closest_glosses.append((label, filtered_labels[closest_index]))
    else:
        closest_glosses.append((label, "No match"))

# Save results to csv file
pd.DataFrame(closest_glosses, columns=["Gloss", "Closest Gloss"]).to_csv("closest_gloss_matches.csv", index=False)
print("Saved closest gloss matches to 'closest_gloss_matches.csv'.")

# Create a t-SNE plot to visualise the embeddings
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(gloss_embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("Phonological Embedding Space (FNN)")
plt.show()
