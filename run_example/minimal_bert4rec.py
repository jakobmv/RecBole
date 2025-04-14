from recbole.quick_start import run_recbole

if __name__ == "__main__":
    # Minimal configuration for BERT4Rec on MovieLens-100k
    config_dict = {
        # Essential BERT4Rec hyperparameters
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 64,
        "inner_size": 256,
        "hidden_dropout_prob": 0.2,
        "attn_dropout_prob": 0.2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "mask_ratio": 0.2,
        "loss_type": "CE",
        # Training config
        "epochs": 1,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        # Dataset config
        "load_col": None,
        "neg_sampling": None,
        "train_neg_sample_args": None,  # Required for CE loss
        "eval_neg_sample_args": None,  # Required for CE loss
        # Data path
        "data_path": "./dataset",
        # Evaluation config
        "metrics": ["Recall", "MRR", "NDCG"],
        "topk": [5, 10, 20],
        "valid_metric": "MRR@10",
        # Data splitting
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        },
    }

    # Run the model
    run_recbole(
        model="BERT4Rec",
        dataset="ml-100k",
        config_dict=config_dict,
        config_file_list=None,
    )
