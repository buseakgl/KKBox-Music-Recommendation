# Music Recommendation System with ALS

This project implements a personalized music recommendation engine using the KKBox Dataset. It leverages collaborative filtering techniques to predict user preferences based on large-scale implicit feedback.

## Key Features

* Matrix Factorization using Alternating Least Squares (ALS) with 32 latent factors
* Processed and cleaned over 7 million user-song interactions
* Dimensionality reduction using PCA and t-SNE for visualization
* Evaluation using Precision@10

## Tech Stack

* Python
* Pandas
* NumPy
* SciPy (Sparse Matrices)
* implicit (ALS model)
* Scikit-learn (PCA, t-SNE)
* Matplotlib

## Dataset Overview

The dataset focuses on implicit user behavior:

* msno: User ID
* song_id: Song ID (anonymized)
* target: 1 (listened), 0 (skipped)

Only positive interactions were used to model actual listening preferences.
This approach reduces noise from skipped tracks and focuses on meaningful user engagement.

## Results and Visualization

The model successfully learned latent relationships between songs.
t-SNE visualization shows that similar songs form distinct clusters in the latent space.
<img width="1158" height="678" alt="image" src="https://github.com/user-attachments/assets/ee3f293a-e72e-4e55-a836-e1c6a5b1d2e6" />

## Performance

Precision@10 = 0.2474


