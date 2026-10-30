#!/bin/sh

#python iql_offline.py
#python iql_offline.py
#python iql_active.py
python iql_active.py "--n_steps" "1000000" "--active_data_max_size" "1000000" "--num_samples" "1000000" "--candidate_size" "50" "--selection_size" "10" "--trajectory_size" "400"
