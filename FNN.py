import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Read in dataset for further processing
df = pd.read_csv("path/to/your/dataset.csv")

# Extract phonological features
feature_columns = df.columns[2:] # Adjust columns to match your dataset

# One-hot encode the categorical values of the phonological features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_data = encoder.fit_transform(df[feature_columns].astype(str))  # Convert to string to avoid errors

# Convert one-hot encoded data to PyTorch tensors
X = torch.tensor(one_hot_data, dtype=torch.float32) 

# Buil Feedforward Neural Network
class FFNEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNEmbedding, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Parameters for model
input_size = X.shape[1]  # Number of features
hidden_size = 256        # Hidden layer size       
output_size = 32         # Resulting embedding dimension

# Initialize model, mean square error loss, and Adam optimiser
model = FFNEmbedding(input_size, hidden_size, output_size)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data into PyTorch DataLoader
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create a training loop with 50 epochs
epochs = 50
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch[0])
        target = model(batch[0].detach())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Extract embeddings
with torch.no_grad():
    embeddings = model(X).numpy()

# Compute pairwise cosine similarity
cosine_sim = cosine_similarity(embeddings)

pairs = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        pairs.append((df.iloc[i]['Gloss 1'], df.iloc[j]['Gloss 1'], cosine_sim[i, j]))

# Sort pairs by similarity score
pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

# Save results to csv file
pd.DataFrame(pairs, columns=['Gloss 1', 'Gloss 2', 'Cosine Similarity']).to_csv("closest_pairs_ffn.csv", index=False)

# Create a t-SNE plot to visualise the embeddings
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("Phonological Embedding Space (FNN)")
plt.show()
