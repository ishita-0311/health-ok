"""Train all 5 disease models sequentially."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import train_heart
import train_diabetes
import train_breast_cancer
import train_ckd
import train_parkinsons


def main():
    print("=== Training Heart Disease models ===")
    train_heart.main()
    print("=== Training Diabetes models ===")
    train_diabetes.main()
    print("=== Training Breast Cancer models ===")
    train_breast_cancer.main()
    print("=== Training CKD models ===")
    train_ckd.main()
    print("=== Training Parkinson's models ===")
    train_parkinsons.main()
    print("=== All models trained ===")


if __name__ == "__main__":
    main()
