# train.py
import sys
import os
from traffic import train_model

def main():
    """
    Train the traffic sign recognition model.
    
    Usage: python train.py data_directory [model.h5]
    """
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train.py data_directory [model.h5]")
    
    data_dir = sys.argv[1]
    model_output = sys.argv[2] if len(sys.argv) == 3 else "traffic.h5"
    
    # Train and save the model
    train_model(data_dir, model_output)


if __name__ == "__main__":
    main()