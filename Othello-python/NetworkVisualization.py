from misc_functions import wipe_dir
from Models import get_dqn_network, initialize_networks
from keras.utils.vis_utils import plot_model
import tensorflow as tf

def plot_all(output_dir):
    (low_network, low_network_target, high_network, high_network_target, high_network_critic, high_network_critic_target) = initialize_networks()
    dqn_network = get_dqn_network()

    models = [low_network, high_network, high_network_critic, dqn_network]
    names = ["low_network", "high_network", "high_network_critic", "dqn_network"]
    
    print("Plotting models to {}...".format(output_dir))

    for (model, name) in zip(models, names):
        plot_model(model, to_file="{}/{}.png".format(output_dir, name), show_shapes=True, show_layer_names=True)

    print("All done!")

if __name__ == "__main__":
    plot_all("./Model_Plots")