import json
import copy

with open('sota_config.json', 'r') as f:
    original_configs = json.load(f)

hyperparameter_variants = {
    "LODA": [
        {"num_bins": 10, "num_random_cuts": 100},
        {"num_bins": 20, "num_random_cuts": 50},
        {"num_bins": 5, "num_random_cuts": 200}
    ],
    
    "xStream": [
        {"num_components": 100, "n_chains": 100, "depth": 25, "window_size": 25},
        {"num_components": 50, "n_chains": 50, "depth": 15, "window_size": 50},
        {"num_components": 200, "n_chains": 200, "depth": 35, "window_size": 10}
    ],
    
    "HSTree": [
        {"window_size": 100, "num_trees": 25, "max_depth": 15},
        {"window_size": 50, "num_trees": 15, "max_depth": 10},
        {"window_size": 200, "num_trees": 35, "max_depth": 20}
    ],
    
    "RSHash": [ 
        {"sampling_points": 1000, "decay": 0.015, "num_components": 100, "num_hash_fns": 1},
        {"sampling_points": 500, "decay": 0.01, "num_components": 50, "num_hash_fns": 2},
        {"sampling_points": 2000, "decay": 0.02, "num_components": 200, "num_hash_fns": 1}
    ],
    
    "IForestASD": [
        {"window_size": 2048},
        {"window_size": 1024},
        {"window_size": 4096}
    ],
    
    "ARCUS": [
        {
            "model_type": "RAPP", "inf_type": "ADP", "seed": 0, "gpu": "0",
            "batch_size": 512, "min_batch_size": 32, "init_epoch": 5,
            "intm_epoch": 1, "hidden_dim": 24, "layer_num": 3,
            "learning_rate": 0.0001, "reliability_thred": 0.95, "similarity_thred": 0.8
        },
        {
            "model_type": "RAPP", "inf_type": "ADP", "seed": 0, "gpu": "0",
            "batch_size": 256, "min_batch_size": 16, "init_epoch": 10,
            "intm_epoch": 2, "hidden_dim": 16, "layer_num": 2,
            "learning_rate": 0.0005, "reliability_thred": 0.90, "similarity_thred": 0.7
        },
        {
            "model_type": "RAPP", "inf_type": "ADP", "seed": 0, "gpu": "0",
            "batch_size": 1024, "min_batch_size": 64, "init_epoch": 3,
            "intm_epoch": 1, "hidden_dim": 32, "layer_num": 4,
            "learning_rate": 0.00005, "reliability_thred": 0.98, "similarity_thred": 0.85
        }
    ]
}

new_configs = []

for config in original_configs:
    algo_name = config["name"]
    
    if algo_name in hyperparameter_variants:
        config_v2 = copy.deepcopy(config)
        config_v2["argument"] = hyperparameter_variants[algo_name][1]
        new_configs.append(config_v2)
        
        config_v3 = copy.deepcopy(config)
        config_v3["argument"] = hyperparameter_variants[algo_name][2]
        new_configs.append(config_v3)

with open('new_hyperparameter_configs.json', 'w') as f:
    json.dump(new_configs, f, indent=4)


for algo in hyperparameter_variants.keys():
    count = len([c for c in new_configs if c["name"] == algo])
    print(f"   {algo}: {count} configs (2 variantes × {count//2} datasets)")