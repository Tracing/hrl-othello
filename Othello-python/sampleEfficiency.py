from scipy.stats import ttest_ind
import csv
import gc
import math
import numpy as np
import subprocess

n = 30
alpha = 0.05
z = 1.96
infile_path = "./sampleEffiencyStatistics.csv"
outfile_path = "./sample_effiency_output.txt"
outfile_data_path = "./sample_effiency_output.csv"

def bonferroni_correction(alpha, k):
    return alpha / k

def generate_latex(outfile_path, N, mean_hiro, stdev_hiro, mean_dqn, stdev_dqn, dqn_hiro_p_value, dqn_hiro_significant):
    #How many decimal places?
    confint_hiro = z * (stdev_hiro / math.sqrt(N))
    confint_dqn = z * (stdev_dqn / math.sqrt(N))

    latex_lines = ["\\begin{center}", 
                   "\\begin{tabular}{c | c | c}",
                   "Algorithm & N & Mean Epochs",
                   "HIRO-A & {} & {:.18f}+-{:.18f}".format(N, mean_hiro, confint_hiro),
                   "DQN & {} & {:.18f}+-{:.18f}".format(N, mean_dqn, confint_dqn),
                   "\\end{tabular}",
                   "\\end{center}",
                   "",
                   "\\begin{center}", 
                   "\\begin{tabular}{c | c | c}",
                   "Algorithm 1 & Algorithm 2 & p value",
                   "HIRO-A & DQN & {:.18f}{}".format(dqn_hiro_p_value, "*" if dqn_hiro_significant else ""),
                   "\\end{tabular}",
                   "\\end{center}"
                   ]
    string = "\n".join(latex_lines)

    with open(outfile_path, "w") as f:
        f.write(string)

def generate_data(outfile_data_path, HIRO_statistics, DQN_statistics):
    lines = ["HIRO_Epochs, DQN_Epochs, HIRO_Score_Vs_Random, DQN_Score_vs_Random, HIRO_Games_Vs_Random, DQN_Games_Vs_Random"]
    for i in range(len(HIRO_statistics)):
        lines.append("{}, {}, {}, {}, {}, {}".format(HIRO_statistics[i][0], DQN_statistics[i][0], 
                                                                 HIRO_statistics[i][1], DQN_statistics[i][1], 
                                                                 HIRO_statistics[i][2], DQN_statistics[i][2]))
    string = "\n".join(lines)

    with open(outfile_data_path, "w") as f:
        f.write(string)

def produce_data():
    HIRO_statistics = []
    DQN_statistics = []
    for seed in range(1, n+1):
        subprocess.check_call(["python3", "sampleEfficiencyGeneration.py", "{}".format(seed), "dqn"])
    for seed in range(1, n+1):
        subprocess.check_call(["python3", "sampleEfficiencyGeneration.py", "{}".format(seed), "hiro"])
    # with open(infile_path, "r") as f:
    #     reader = csv.reader(f, delimiter=',')
    #     for (x1, x2, x3, x4, x5, x6) in reader:
    #         HIRO_statistics.append((int(x1), float(x3), float(x5)))
    #         DQN_statistics.append((int(x2), float(x4), float(x6)))
    return (HIRO_statistics, DQN_statistics)

def main():
    (HIRO_statistics, DQN_statistics) = produce_data()

    # print("alpha = {:.4f}".format(alpha))

    # print("HIRO statistics")
    # print(HIRO_statistics)

    # print("DQN statistics")
    # print(DQN_statistics)

    # HIRO_data = [x[0] for x in HIRO_statistics]
    # DQN_data = [x[0] for x in DQN_statistics]

    # hiro_dqn_p_value = ttest_ind(HIRO_data, DQN_data, alternative='less')[1]

    # hiro_dqn_p_value_significant = hiro_dqn_p_value <= alpha

    # print("HIRO - dqn p-value = {:.5f}".format(hiro_dqn_p_value))

    # if hiro_dqn_p_value_significant:
    #     print("HIRO - dqn is significant")
    # else:
    #     print("HIRO - dqn is not significant")

    # print("HIRO Epochs Mean = {:.5f}".format(np.mean(HIRO_data)))
    # print("HIRO Epochs Stdev = {:.5f}".format(np.std(HIRO_data)))

    # print("dqn Epochs Mean = {:.5f}".format(np.mean(DQN_data)))
    # print("dqn Epochs Stdev = {:.5f}".format(np.std(DQN_data)))
    

    # print("Generating latex code to {}...".format(outfile_path))

    # generate_latex(outfile_path, n, np.mean(HIRO_data), 
    #                      np.std(HIRO_data), np.mean(DQN_data), 
    #                      np.std(DQN_data), hiro_dqn_p_value, 
    #                      hiro_dqn_p_value_significant)
    
    # print("Saving data to {}...".format(outfile_data_path))

    # generate_data(outfile_data_path, HIRO_statistics, DQN_statistics)

    # print("All done!")

if __name__ == "__main__":
    main()
