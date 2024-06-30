# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 01:54:43 2024

@author: user
"""

import numpy as np
from pykalman import KalmanFilter
#pip install pykalman

# Generate synthetic data for demonstration
np.random.seed(42)  # Seed for reproducibility

# Parameters
n = 4  # Number of states
m = 2  # Number of observations
t = 50  # Number of time points

# True system dynamics (for data generation, not known in real applications)
true_A = np.array([[0.5, 0.1, 0, 0],
                   [0.1, 0.5, 0, 0],
                   [0, 0, 0.3, 0.1],
                   [0, 0, 0.1, 0.3]])
true_C = np.array([[1, 0, 1, 0],
                   [0, 1, 0, 1]])
true_Q = 0.1 * np.eye(n)  # Process noise covariance
true_R = 0.1 * np.eye(m)  # Observation noise covariance

# Control input (random)
u = np.random.randn(t, n)

# State and observation generation
x = np.zeros((t, n))
y = np.zeros((t, m))
x[0] = np.random.multivariate_normal(np.zeros(n), true_Q)  # Initial state
for i in range(1, t):
    x[i] = np.dot(true_A, x[i-1]) + np.dot(true_Q, np.random.randn(n))
    y[i] = np.dot(true_C, x[i]) + np.dot(true_R, np.random.randn(m))

# Initialize the Kalman Filter with guesses
initial_A = 0.5 * np.eye(n)
initial_C = 0.5 * np.eye(m, n)
initial_Q = 0.5 * np.eye(n)
initial_R = 0.5 * np.eye(m)

kf = KalmanFilter(transition_matrices=initial_A, observation_matrices=initial_C,
                  transition_covariance=initial_Q, observation_covariance=initial_R)

# EM algorithm to estimate A, C, Q, R
kf = kf.em(y, n_iter=5)

# Estimated parameters
A_est = kf.transition_matrices
C_est = kf.observation_matrices
Q_est = kf.transition_covariance
R_est = kf.observation_covariance

print("Estimated A:\n", A_est)
print("Estimated C:\n", C_est)
print("Estimated Q:\n", Q_est)
print("Estimated R:\n", R_est)