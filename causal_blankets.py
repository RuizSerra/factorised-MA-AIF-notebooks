from collections import Counter
import numpy as np
from scipy.stats import entropy

def bin2dec(bin_array):
    '''
    Convert a binary array to decimal.
    Args:
        bin_array (np.ndarray): A 2D binary array where each row is a binary number.
    Returns:
        np.ndarray: A 1D array of decimal numbers.
    '''
    # Ensure the input is a numpy array
    bin_array = np.asarray(bin_array)
    # Check if the input is binary
    if not np.all(np.isin(bin_array, [0, 1])):
        raise ValueError("Input array must be binary (0s and 1s only).")
    # Check if the input is 2D
    if bin_array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    # Convert binary to decimal
    return np.dot(bin_array, 2**np.arange(bin_array.shape[1])[::-1]).astype(int)

def estimate_conditional_probabilities(X, Y, L=1):
    '''
    Estimate p(y_{t+1} | x_{t-L:t}, y_{t-L:t})

    Returns:
        dist_support (np.ndarray): The support for the probability distribution p(y | X_L, Y_L).
        X_sub_flat (np.ndarray): The flattened subsequences of X (shape: (T-L+1, L*n)).
        Y_sub_flat (np.ndarray): The flattened subsequences of Y (shape: (T-L+1, L*m)).
        conditional_probs (dict): The conditional probabilities p(y | X_L, Y_L) for each subsequence X_L.
    '''

    T = X.shape[0]
    n = X.shape[1]
    m = Y.shape[1]
        
    # Generate subsequences
    X_subsequences = np.array([X[t-L:t] for t in range(L, len(X))])  # shape: (T-L+1, L, n)
    Y_subsequences = np.array([Y[t-L:t] for t in range(L, len(Y))])  # shape: (T-L+1, L, m)

    # Flatten subsequences for input to the model
    X_sub_flat = X_subsequences.reshape(X_subsequences.shape[0], -1)  # shape: (T-L+1, L*n)
    Y_sub_flat = Y_subsequences.reshape(Y_subsequences.shape[0], -1)  # shape: (T-L+1, L*m)

    # Combine X_sub and Y_sub as features
    features = np.hstack((X_sub_flat, Y_sub_flat))  # { X_{t-L:t}, Y_{t-L:t} }  shape: (T-L+1, L*n + L*m)

    # Target is the current state Y
    targets = Y[L:]  # { Y_{t} }  shape: (T-L+1, m)

    # Convert features and targets to tuples for counting
    features_tuples = [tuple(f) for f in features]
    target_tuples = [tuple(t) for t in targets]

    # Count occurrences of each feature-target pair, and of each feature
    joint_counts = Counter(zip(features_tuples, target_tuples))
    feature_counts = Counter(features_tuples)

    # Generate the support for the probability distribution
    dist_support = (
        np.array([list(map(int, f"{i:0{m}b}")) for i in range(2**m)])
    )  # shape: (2^m, m) e.g. [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], ...] for m=3

    # Estimate conditional probabilities
    conditional_probs = {
        tuple(X_L): {}  # X_L is a single length-L subsequence of X
        for X_L in np.unique(X_sub_flat, axis=0)
    }
    for (feature, target), joint_count in joint_counts.items():
        X_L = tuple(feature[:L*n])
        Y_L = tuple(feature[L*n:])
        
        # Create a new entry in the dictionary if it doesn't exist
        if conditional_probs[X_L].get(Y_L) is None:
            conditional_probs[X_L][Y_L] = np.zeros(dist_support.shape[0])
        
        # Update the conditional probability for the target
        target_index = np.where((dist_support == target).all(axis=1))[0][0]
        conditional_probs[X_L][Y_L][target_index] = joint_count / feature_counts[feature]

    return dist_support, X_sub_flat, Y_sub_flat, conditional_probs

def get_minimal_DBaSS(X, Y, epsilon=1e-3, L=1):
    """
    Get the minimal D-BaSS causal states for a given set of conditional probabilities.

    From Rosas2020CausalBlankets:
    
        Definition 1: Given two stochastic processes X, Y, 
        a process U is a dynamical Bayesian Suficient Statistic (D-BaSS) of X w.r.t. Y if
        for all t, the following hold:
            1. Precedence: there exists a function F such that U_t = F(X_t)
            2. Sufficiency: Y_t+1 is independent of X_t given (U_t, Y_t)
        M is a minimal D-BaSS if, for any D-BaSS U, there exists a function f such that f(U_t) = M_t

        Theorem 1: Given stochastic processes X, Y, the minimal D-BaSS of X w.r.t. Y 
        corresponds to the partition of past trajectories x_t induced by the following equivalence relationship:
            x_t ~ x'_t iff p(y_{t+1} | x_t, y_t) = p(y_{t+1} | x'_t, y_t) for all y_t and y_{t+1}

    Args:
        X (np.ndarray): The input data (shape: (T, n)).
        Y (np.ndarray): The output data (shape: (T, m)).
        epsilon (float): The threshold for KL divergence to determine if two distributions are similar.
        L (int): The length of the subsequences to consider.
    """
    
    # Get p(y_{t+1} | x_t, y_t)
    _, X_sub_flat, Y_sub_flat, conditional_probs = estimate_conditional_probabilities(X, Y, L=L)

    causal_states = []  # The list of causal states grows as new states are "discovered"
    for X_L in X_sub_flat:
        # Check what causal state each X_L belongs to
        X_L = tuple(X_L)
        for causal_state in causal_states:
            match = True
            # Check if the current X_L matches any existing causal state (for all Y_L and y)
            for Y_L in conditional_probs[X_L].keys():
                support_length = conditional_probs[X_L][Y_L].shape[0]
                target_distribution = (  # p(y | X_L, Y_L)
                    causal_state['distributions']
                    .get(
                        Y_L, # Retrieve the "canonical" distribution for Y_L under this causal state
                        np.ones(support_length) / support_length  # If Y_L is not found, use MaxEnt distribution (FIXME?)
                    )
                )  
                KL_divergence = entropy(
                    conditional_probs[X_L][Y_L], 
                    target_distribution
                )
                if KL_divergence > epsilon:
                    match = False
                    break  # No match, move on to the next causal state
            if match:  # Distributions for all Y_L matched
                # Add the current X_L to the members of the matching causal state
                causal_state['members'].add(X_L)
                # Update the canonical distribution for the matching causal state
                # TODO: is this necessary?
                causal_state['distributions'].update(conditional_probs[X_L])
                break
        # Having exhausted all causal states, if no match was found, create a new one
        if len(causal_states) == 0 or not match:
            causal_states.append({
                'members': {X_L},
                'distributions': conditional_probs[X_L],
                'label': len(causal_states),  # Label the new causal state
            })

    return causal_states, X_sub_flat

def get_causal_state_timeseries(causal_states_inventory, subsequences_timeseries):
    """
    Get the causal state time series from the subsequences time series.

    Args:
        causal_states_inventory (list): list of causal state definitions (dicts).
        subsequences_timeseries (np.ndarray): timeseries of subsequences of length L (shape: (T, L*n)).
    """

    cs_timeseries = np.empty(subsequences_timeseries.shape[0], dtype=int)
    for t, X_L in enumerate(subsequences_timeseries):
        X_L = tuple(X_L)
        for causal_state in causal_states_inventory:
            if X_L in causal_state['members']:
                cs_timeseries[t] = causal_state['label']
                break
    return cs_timeseries