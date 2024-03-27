from DQNTrain import dqnTrain
from HIROTrain import HIROTrain
import sys

def train_HIRO(seed):
    x = HIROTrain("HIRO_{}".format(seed), seed)
    return x

def train_DQN(seed):
    x = dqnTrain("DQN_{}".format(seed), seed)
    return x

def produce_data(seed, is_hiro):
    if is_hiro:
        train_HIRO(seed)
    else:
        train_DQN(seed)
    
if __name__ == "__main__":
    seed = int(sys.argv[1])
    assert sys.argv[2] in ["dqn", "hiro"]
    is_hiro = sys.argv[2] == "hiro"
    outfile = "./sampleEffiencyStatistics.csv"

    produce_data(seed, is_hiro)