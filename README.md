# EE660 Project
Affiliated course: USC EE660, Machine learning: Foundations & Methods, given in 19 fall semester by Prof. Jenkins

Used Dataset: Breast Cancer Wisconsin (Diagnostic)

Problem Type: Binary Classification

Methods Involved:
  - Preprocessing & Visualization:
    - Scatter / Box plots features with the top-5 CV
    - Standardization
    - Feature correlation heat matrix
    - Per-feature class distribution plotting
    - Linear / Kernel PCA
  - Part1, Finding Best Performance Models:
    - K-nearest Neighbor
    - AdaBoost & Boosting Forest
    - Elastic Net Logistic Regression
    - K Means Clustering + Logistic Regression
  - Part2, Experimental Explorations:
    - Monte Carlo Semi-supervised l1 support vector machine
    - Spectral Clustering

Code Organization:
  - Main.py:      Highest level script for generating results
  - dataGen.py:   Class "dataGenerator" for data reading, preprocessing, visualization & data fetching
  - modules.py:   Class "models" containing all training & testing process of models used in this project
  - utils.py:     Class "util" for all graph plotting, dataset generating / fetching for models, result / log printing.

Dependencies:
  Python:         3.7.4
  
  Scikit-learn:   0.21.3
  
  Pandas:         0.25.1
  
  Seaborn:        0.9.0
  
  Other:          numpy, datetime, shutil, os, statistics, matplotlib.pyplot, random
