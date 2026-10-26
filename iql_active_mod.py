from utils_common import *
from utils_plotting import plot_states, plot_states_uncertainties, plot_collection, plot_buffer_evolution, plot_rewards
from utils_plotting import get_active_rewards
from utils_dataset import reduce_dataset
from utils_uncertainty import UncertaintyModel
from utils_uncertainty import Collector
from utils_dataset import dataset_from_episode_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="PyFlyt/QuadX-Hover-v4")
    parser.add_argument("--data", type=str, default="dataset-14-0-20-v3")

    #IQL Parameters
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--expectile", default=0.9, type=float)
    parser.add_argument("--weight_temp", default=10.0, type=float)
    parser.add_argument("--max_weight", default=100.0, type=float)


    #Uncertainty Model Parameters
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--uncertainty_lr", default=3e-4, type=float)
    parser.add_argument("--num_models", default=5, type=int)
    parser.add_argument("--width", default=256, type=int)
    parser.add_argument("--depth", default=5, type=int)

    #Training Parameters
    parser.add_argument("--n_steps", default=50000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=10000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    #Active Parameters
    parser.add_argument("--active_epochs", default=10, type=int)
    parser.add_argument("--active_data_max_size", default=100000, type=int)

    parser.add_argument("--num_samples", default=5000, type=int)
    parser.add_argument("--candidate_size", default=20, type=int)
    parser.add_argument("--selection_size", default=2, type=int)
    parser.add_argument("--trajectory_size", default=200, type=int)

    parser.add_argument("--collection_method", default="active", type=str)
    parser.add_argument("--collection_policy", default="uncertain-dataset", type=str)
    parser.add_argument("--collection_epsilon", default=0.2, type=float)
    parser.add_argument("--active_epsilon", default=0.8, type=float)

    parser.add_argument("--es", default=False, action="store_true")
    parser.add_argument("--es_threshold", default=0.2, type=float)
    parser.add_argument("--es_percentile", default=60, type=float)
    parser.add_argument("--up_percentile", default=90, type=float)
    parser.add_argument("--pn_noise", default=0.05, type=float)

    #Extra arguments
    parser.add_argument("--n_trial", default=100, type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--show_progress", action="store_true", default=False)

    args = parser.parse_args()
    return args

def main(args):
    logdir = f"IQL_ACTIVE/{args.env}/"
    logdir += f"native" if args.data == "" else f"{args.data}/"

    experiment_name = "N_"

    if args.collection_method == "active":
        experiment_name += f"a{args.active_epsilon}"

    #RANDOM: Pick the first states
    elif args.collection_method == 'random':
        experiment_name += "r"

    #INITIAL: Pick the initial states
    elif args.collection_method == 'initial':
        experiment_name += "i"
    else:
        raise ValueError

    if 'random' in args.collection_policy:
        experiment_name += "r"

    elif 'policy' in args.collection_policy:
        experiment_name += 'p'

    elif 'uncertain' in args.collection_policy:
        experiment_name += f"u{args.collection_epsilon}_d{args.depth}_w{args.width}"
    elif 'up' in args.collection_policy:
        experiment_name += f"up({args.up_percentile},{args.es_percentile})_d{args.depth}_w{args.width}"
    elif 'pn' in args.collection_policy:
        experiment_name += f"pn{args.pn_noise}_eps{args.collection_epsilon}_d{args.depth}_w{args.width}"
    else:
        raise ValueError

    experiment_name += f"_t{args.trajectory_size}_cs{args.candidate_size}_ss{args.selection_size}"
    
    if args.es:
        experiment_name += f"_es"

    experiment_name += f"_{args.seed}"

    experiment_root = f"{logdir}/{experiment_name}/"

    #Create Directory
    setup_directory(experiment_root)

    seed_everything(args.seed)

    dataset, env = d3rlpy.datasets.get_minari("QuadX-Hover-v4/dataset-v0",
            action_space=d3rlpy.ActionSpace.CONTINUOUS)
    active_states = torch.from_numpy(dataset.observations).float()

    env.seed(args.seed)
