import os
import os.path
import shutil

def get_paths(root):
    filenames = []
    filepaths = []

    for dir in os.listdir(root):
        filenames.append("{}.csv".format(dir))
        filepaths.append("{}/{}/data.csv".format(root, dir))

    return (filenames, filepaths)

def create_new_files(filenames, filepaths, output_dir):
    for (filename, filepath) in zip(filenames, filepaths):
        output_filepath = "{}/{}".format(output_dir, filename)
        shutil.copy(filepath, output_filepath)

def to_r_list(output_dir, n_seeds, output_file):
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]
    experiments = {}
    string_list = []

    for env_name in env_names:
        for train_low_name in ["low_train", "only_high", "HighFromScratch", "FromScratch", "ZeroShot", "LinearFreeze", "LinearFreezeHighOnly"]:
            experiment_name = "Transfer-{}-{}".format(env_name, train_low_name)
            
            experiments[experiment_name] = []
            
            for seed in range(1, n_seeds+1):
                experiments[experiment_name].append("\"{}/{}-{}.csv\"".format(output_dir, experiment_name, seed))
    
    for experiment_name in experiments:
        s = ", ".join(experiments[experiment_name])
        s = "a <- c({})".format(s)
        string_list.append(s)
    
    output = "\n\n".join(string_list)

    with open(output_file, "w") as f:
        f.write(output)

def main():
    root = "evaluation"
    output_dir = "transferability_experiment_data"
    r_commands_output_file = "files.r"

    (filenames, filepaths) = get_paths(root)
    create_new_files(filenames, filepaths, output_dir)
    to_r_list(output_dir, 10, r_commands_output_file)

if __name__ == "__main__":
    main()