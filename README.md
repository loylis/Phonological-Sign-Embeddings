# Phonological-Sign-Embeddings
This repository contains code used to train the phonological sign embeddings in the "The Phonological Space of Sign Languages: Embeddings’ Ability to Capture Human-defined Linguistic Constraints without Instruction" paper.

# Description

There is growing interest in using phonetic word embeddings as a tool for linguistic research. Although the literature on using embeddings for spoken language research is growing rapidly, the use of embeddings for sign language research remains largely unexplored. In this study on phonological similarity in sign languages, we investigate how three different embedding methods – word2vec, one-hot-encoding + truncated singular value decomposition (OHE-SVD), and a feedforward neural network (FNN) – compare to a rule-based approach based on linguistic constraints. A dataset of 1,880 signs in Kenyan Sign Language (KSL) is analysed and examined for minimal pairs. We conclude word2vec is not fruitful for sign language data. However the OHE-SVD and FNN approaches both successfully identify both minimal pairs and homophones in the KSL data as measured by cosine similarity scores. Furthermore, significant differences in average similarity scores according to phonological feature unveil valuable insights into the functional load of the number of hands and handshape.

This repository contains three different Python scripts: 

- word2vec.py => to train embeddings using the word2vec architecture
- ohesvd.py => to train embeddings using one-hot-encoding + truncated singular value decomposition
- FNN.py => to train embeddings using a feedforward neural network 

# Getting Started

## Dependencies

Before running the code in this repository, the following libraries need to be installed:

- Pandas 2.2.3
- Numpy 1.26.4
- Scikit-Learn 1.6.1
- Matplotlib 3.9.4
- Seaborn 0.13.2
- Gensim 4.3.3
- Pytorch 2.6.0

All scripts were originally written to be run in Visual Studio Code with Python 3.9.6. Therefore, to be able to run the scripts from the command line, the scripts need to be slightly adjusted.

*** Disclaimer ***

The scripts in this repository have been cleaned and optimized with the help of the inbuilt GitHub Copilot functionality in Visual Studio Code. All ideas and the general code architecture are the author's own and Copilot was only used to improve, not to generate.

## word2vec.py

Employing the Gensim Python library to train the word2vec embeddings, using the CBOW architecture with a context window of 5, a minimum count of 1 to include all feature instances, and the default learning rate of 0.025. The default values are used for all other hyperparameters.

## ohesvd.py

This is a reproduction of Martinez del Rio et al.’s (2022) approach, referred to as OHE-SVD. This script creates phonological sign embeddings by one-hot-encoding a vector consisting of the values for a phonological annotation of each sign in a database. I use the OneHotEncoder function in the Scikit-Learn library and then perform dimensionality reduction using truncated SVD on the one-hot encoded vectors, employing the TruncatedSVD function in the same library.

## FNN.py

The same process as described above is used to one-hot encode a phonological feature string before then converting the encoded strings to Pytorch tensors. The Pytorch Python library is used to build a feedforward neural network (FNN) with one hidden layer of the size 256, a rectified linear unit (ReLU) activation function, training on 50 epochs. The model is evaluated using a mean squared error (MSE) loss function and the Adam optimiser to optimise its gradient descent.

# Authors

Lisa Loy
lisa.loy@uni-hamburg.de

# License

This project is licensed under the Creative Commons Attribution 4.0 License - see the LICENSE.md file for detail.
