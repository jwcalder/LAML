# Linear Algebra, Data Science, and Machine Learning Textbook

This is the main website for electronic material for the textbook

J. Calder, P. J. Olver. [Linear Algebra, Data Science, and Machine Learning](https://), Springer 2025.

## Student Solutions Manual and Errata

The student solutions manual and errata for the book can be found at the links below.

* [Student Solutions Manual](https://)
* [Errata](https://)

## Python Notebooks

Below are descriptions of, and links to, the Python notebooks from the main body of the text. There are additional homework solution notebooks that are linked in the student solution manual. All notebooks are stored in this GitHub repository, but clicking on the link below will conveniently open the notebook in Google Colab. 

### Introductory Notebooks
* Basic introduction to the Python programming language:
        [IntroPython](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/IntroPython.ipynb)
* Basic introduction to the Python programming language with solutions to exercises:
        [IntroPythonSol](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/IntroPythonSol.ipynb) 
* Basic introduction to the NumPy package:
        [IntroNumpy](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/IntroNumpy.ipynb)
* Basic introduction to the NumPy package with solutions to exercises:
        [IntroNumpySol](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/IntroNumpySol.ipynb)
* An overview of more advanced features of NumPy:
        [AdvancedNumpy](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/AdvancedNumpy.ipynb)
* An overview of more advanced features of NumPy:
        [AdvancedNumpySol](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/AdvancedNumpySol.ipynb)
* Basic introductory notebook showing how to solve linear systems and eigenvalue problems in Numpy (and Scipy):
        [NumpySolvers](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Intro/NumpySolvers.ipynb) 
* Basic introduction to the Pandas package:
        [IntroPandas](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/IntroML/IntroPandas.ipynb)
* Basic introduction to the PyTorch package, which is used for deep learning:
        [IntroPyTorch](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/NeuralNetworks/IntroPyTorch.ipynb)

### QR Factorization (Chapter 4)
* [QRFactorization](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Homework/QRFactorization.ipynb): Notebook on QR factorization and numerical instabilities (Section 4.7).

### Eigenvalues and Singular Values (Chapter 5)
* [NumericalComputationofEigenvalues](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Homework/NumericalComputationofEigenvalues.ipynb): Numerical computation of eigenvalues and eigenvectors using the power method and orthogonal iteration (Section 5.6).

### Optimization (Chapters 6 and 11)
* [GradientDescent](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Optimization/GradientDescent.ipynb): Gradient descent in NumPy from scratch (Section 6.4).
* [NewtonsMethod](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Optimization/NewtonsMethod.ipynb): Newton's method in NumPy (Section 6.10).
* [StochasticGradientDescent](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/Optimization/StochasticGradientDescent.ipynb): Some basic toy examples of stochastic gradient descent (Section 11.5).

### Introduction to Machine Learning (Chapter 7)
* [LinearRegression](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/IntroML/LinearRegression.ipynb): Basic example of ridge regression in NumPy (Section 7.2).
* [SupportVectorMachines](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/IntroML/SupportVectorMachines.ipynb): Examples of how to train a support vector machine using sklearn (Section 7.3).
* [kNearestNeighbor](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/IntroML/kNearestNeighbor.ipynb): Examples of how to train a k-nearest neighbors classifier using sklearn (Section 7.4).
* [kMeans](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/IntroML/kMeans.ipynb): Code demonstrating the details of the k-means algorithm using NumPy (Section 7.5).

### Principal Component Analysis (Chapter 8)
* [IntroPCA](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/PrincipalComponentAnalysis/IntroPCA.ipynb): Computation of principal components in NumPy and SciPy and applications (Section 8.1).
* [PCACompression](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/PrincipalComponentAnalysis/PCACompression.ipynb): Code examples of PCA-based image compression (Section 8.3).
* [LDA](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/PrincipalComponentAnalysis/LDA.ipynb): Code demonstrating how to compute the linear discriminant analysis (LDA) embedding in NumPy and SciPy (Section 8.4).
* [MDS](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/PrincipalComponentAnalysis/MDS.ipynb): Code demonstrating how to multidimensional scaling (MDS) in NumPy and SciPy  (Section 8.5).

### Graph-Based Learning (Chapter 9)
* [IntroToGraphs](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/IntroToGraphs.ipynb): Introduction to graphs and how to load and display them with the [GraphLearning](https://github.com/jwcalder/GraphLearning) Python package (Section 9.1).
* [BinarySpectralClustering](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/BinarySpectralClustering.ipynb): Binary spectral clustering using the [GraphLearning](https://github.com/jwcalder/GraphLearning) package (Section 9.4).
* [ShortestPaths](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/ShortestPaths.ipynb): Overview of how to compute shortest paths and their lengths on graphs using the [GraphLearning](https://github.com/jwcalder/GraphLearning) package (Section 9.5).
* [PageRank](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/PageRank.ipynb): Computation of the PageRank vector and applications  (Section 9.6).
* [SpectralEmbeddings](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/SpectralEmbeddings.ipynb): Demo of spectral embeddings and spectral clustering using the [GraphLearning](https://github.com/jwcalder/GraphLearning) package (Section 9.7).
* [tSNE](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/tSNE.ipynb): Implementation of the t-stochastic neighbor embedding (t-SNE) in NumPy, with example applications (Section 9.8).
* [GraphBasedSSL](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/GraphBasedSSL.ipynb): Demonstration of graph-based semi-supervised learning with the [GraphLearning](https://github.com/jwcalder/GraphLearning) package (Section 9.9).
* [DiscreteFourierTransform](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/GraphBasedLearning/DiscreteFourierTransform.ipynb): Implementation of the fast Fourier transform in NumPy and applications to signal denoising (Section 9.10).

### Neural Networks (Chapter 10)
* [FullyConnectedNeuralNetworks](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/NeuralNetworks/FullyConnectedNeuralNetworks.ipynb): Basic examples of training fully connected neural networks in PyTorch, and applications to synthetic and real data (Section 10.1).
* [ConvolutionalNeuralNetworks](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/NeuralNetworks/ConvolutionalNeuralNetworks.ipynb): Basic examples of training convolutional neural networks in PyTorch, and applications classifying MNIST digits (Section 10.3)
* [GraphConvolutionalNeuralNetworks](https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/NeuralNetworks/GraphConvolutionalNeuralNetworks.ipynb): Basic example of training a graph convolutional neural network for node classification in PyTorch, and application to PubMed academic paper classification. (Section 10.4)


