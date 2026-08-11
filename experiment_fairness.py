
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def experiment_fairness():
    print("Running Fairness Disparate Impact Experiment (Idea 34)...")
    
    # 1. Generate Biased Data
    # Feature X1 (Skills) -> True Predictor of Y (Hiring)
    # Feature S (Gender) -> Irrelevant to Y, but correlated with X1 (Systemic Bias)
    
    np.random.seed(42)
    n = 1000
    
    S = np.random.randint(0, 2, n) # 0 or 1
    
    # X1 depends on S (Bias in input features)
    # Group 1 has mean 0.6, Group 0 has mean 0.4
    X1 = np.random.normal(0.4 + 0.2*S, 0.1) 
    
    # Y depends ONLY on X1 (Skills)
    # Threshold 0.5
    y = (X1 > 0.5).astype(int)
    
    X = np.stack([X1, S], axis=1)
    
    # 2. Train Standard CART
    # It might use S if it helps splitting, or just use X1.
    # Even if it uses X1, it will inherit the bias.
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)
    
    # 3. Measure Disparate Impact
    y_pred = clf.predict(X)
    
    p_1_given_s1 = np.mean(y_pred[S==1])
    p_1_given_s0 = np.mean(y_pred[S==0])
    
    di = p_1_given_s0 / p_1_given_s1 if p_1_given_s1 > 0 else 0
    
    print(f"Selection Rate (Group 1): {p_1_given_s1:.2%}")
    print(f"Selection Rate (Group 0): {p_1_given_s0:.2%}")
    print(f"Disparate Impact (Ratio): {di:.2f}")
    
    # Standard legal threshold is 0.8 (Four-Fifths Rule)
    if di < 0.8:
        print("\n>> VIOLATION: Disparate Impact < 0.8. The model is legally discriminatory.")
        print("Reason: The model accurately learned the bias present in X1.")
        print("Idea 34 (Fairness-Certifiable Trees) would enforce DI > 0.8 as a HARD CONSTRAINT during training,")
        print("forcing the solver to find a suboptimal-accuracy but fair split.")

if __name__ == "__main__":
    experiment_fairness()
