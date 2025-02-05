# CS303 Project3 Report

## 1. Introduction

The project aims to develop a **Knowledge Graph-based Recommender System (KGRS)** to enhance the accuracy and explainability of recommendations. The system uses historical user-item interactions and leverages a **Knowledge Graph (KG)** to better understand the relationships between users, items, and their attributes, thereby providing more personalized recommendations.

## 2. Preliminary Problem Formulation

The goal is to design a score function \( f(u,w) \), which predicts the level of interest a user \( u \) has in an unseen item \( w \). The project uses a dataset of user-item interactions \( Y_{train} \) and a KG \( G=(V,E) \), where:

- \( Y_{train} \): User-item interaction records, indicating whether a user is interested in an item (binary values).
- \( G=(V,E) \): A knowledge graph, with entities \( V \) (users and items) and relationships \( E \).

## 3. Evaluation Metrics

- **AUC (Area Under Curve)**: Evaluates the model's ability to distinguish between positive and negative interactions (CTR prediction task).
- **nDCG@k**: Evaluates the quality of the top-k recommended items, emphasizing relevance and ranking (Top-k recommendation task).

## 4. Methodology

### General Workflow:

1. **Data Preprocessing**: The interaction records and KG data are processed for model training.
2. **KG Embedding**: The entities and relations are represented in a lower-dimensional space.
3. **Model Training**: A model is trained to predict user-item interaction scores using the processed data and KG embeddings.
4. **Evaluation**: The trained model is evaluated using AUC and nDCG@k.

### Algorithm/Model Design:

- **TransE** or **ComplEx** could be used for KG embeddings.
- **Collaborative Filtering** or **Neural Collaborative Filtering** can be employed for recommendation.

## 5. Experiments

### Task 1: CTR Prediction (AUC)

- **Test Flow**: Split the test dataset into positive and negative samples, predict scores, and calculate the AUC score.

### Task 2: Top-k Recommendation (nDCG@5)

- **Test Flow**: For each user, predict scores for all items, rank them, and calculate nDCG@5.

## 6. Experimental Results

- **AUC Score**: **0.702** - This indicates moderate performance in distinguishing between positive and negative samples.
- **nDCG@5 Score**: **0.154** - This suggests that the top-5 recommended items have limited relevance to the user's preferences.
- **Model Configuration**:
  - **Batch size**: 256
  - **Eval batch size**: 1024
  - **Negative sampling rate**: 2
  - **Embedding dimension**: 128
  - **Learning rate**: 0.005
  - **Weight decay**: 0.003
  - **Epochs**: 40
  - **Margin**: 15
  - **L1 regularization**: Disabled

![Model Configuration](media/image1.png){width="5.764583333333333in" height="1.5375in"}

## 7. Analysis of Experimental Results

- The AUC of **0.702** indicates that the system is somewhat effective at predicting user-item interactions. However, there's room for improvement, particularly with the top-k recommendations.
- The **nDCG@5 score** of **0.154** shows that, although the model is making reasonable predictions, the quality of the top-5 recommendations can be improved. This suggests that refining the model or using more advanced techniques could increase the relevance of recommended items.
- The chosen hyperparameters (e.g., embedding dimension of 128, learning rate of 0.005) seem to be a reasonable starting point, though further experimentation could explore different values to optimize performance.

## 8. Conclusion and Future Work

The Knowledge Graph-based Recommender System has shown moderate performance, with a solid foundation for making personalized recommendations. However, there is significant room for enhancement. Future work could focus on:

- **Advanced KG embedding techniques** (e.g., using more sophisticated models like ComplEx or RotatE).
- **Hybrid models** that combine KG embeddings with user-item interaction data more effectively.
- Incorporating additional **user metadata** (e.g., demographics or browsing behavior) to improve personalization.

By refining these aspects, the recommendation system could potentially achieve higher AUC and nDCG scores, leading to better user experience and more accurate recommendations.
