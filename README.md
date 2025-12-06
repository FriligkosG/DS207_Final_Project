# DS 207 Final Project – Predicting ATP Tennis Match Outcomes

**Team:**  
- Georgios Friligkos – georgios.friligkos@berkeley.edu  

## Project Overview

We predict ATP tennis match outcomes using machine learning models on the open Jeff Sackmann tennis dataset.  
We compare a **logistic regression baseline** against **tree-based ensembles (Random Forest, XGBoost)** and a **simple neural network**, and analyze which factors are most influential in determining match outcomes.

## Data

- Source: Jeff Sackmann's ATP Tennis data (1968–2024) from GitHub.  
- We use tour-level matches from 2000–2023.  
- Main variables: surface, tournament level, round, best-of, player rankings and ages.

## Methodology

1. **Preprocessing**
   - Load yearly `atp_matches_YYYY.csv` files and concatenate.
   - Drop matches without ranking information.
   - Build a pairwise dataset in “Player1 vs Player2” format with a random orientation per match.
   - Engineer features such as ranking and age differences, and one-hot encode categorical variables.

2. **Train / Validation / Test Split**
   - Time-based split:
     - Train: ≤ 2016
     - Validation: 2017–2018
     - Test: 2019–2023

3. **Models**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - Feed-forward Neural Network (TensorFlow/Keras)

4. **Evaluation**
   - Accuracy, F1-score, ROC-AUC on validation and test sets.
   - ROC curves on the test set.
   - Feature importances (Random Forest) and coefficients (Logistic Regression).

## How to Run

```bash
git clone https://github.com/FriligkosG/DS207_Final_Project.git
cd DS207_Final_Project

# create env if you want
# conda create -n ds207 python=3.11
# conda activate ds207

pip install -r requirements.txt

# open the notebook
jupyter notebook notebooks/01_tennis_match_prediction.ipynb
