import sys
import subprocess
import time

def main():
    for seed in range(1, 11):
        for function in ["fine_tune", "fine_tune_high_only", "freeze_train_high_only", "freeze_train", "zero_shot", "from_scratch", "from_scratch_high_only"]:
            subprocess.check_call(["python3", "transferability_train.py", function, "{}".format(seed), "4000"])

if __name__ == "__main__":
    main()
