{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "data/SST-2/train.tsv",
    "validation_data_path": "data/SST-2/dev.tsv",
    "model": {
        "type": "base_lstm_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256,
                    "trainable":true
                }
            }
        },
        "encoder":{
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 100,
            "num_layers": 2,
            "bidirectional": true,
            "dropout": 0.5,
        },
        "feedforward":{
          "input_dim":200,
          "num_layers":2,
          "hidden_dims":[200, 64],
          "activations":"relu",
          "dropout":0.5,
        },
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 100
    },
    "distributed": {
        "cuda_devices": [0, 1],
    }
}
