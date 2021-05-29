local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/SST-2/train.tsv",
    "validation_data_path": "data/SST-2/dev.tsv",
    "model": {
        "type": "lstm_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder":{
            "type": "lstm",
            "input_size": 768,
            "hidden_size": 256,
            "num_layers": 2,
            "bidirectional": true,
            "dropout": 0.5,
        },
        "feedforward":{
          "input_dim":512,
          "num_layers":2,
          "hidden_dims":[256, 128],
          "activations":"relu",
          "dropout":0.5,
        },
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    },
    "distributed": {
        "cuda_devices": [0, 1],
    }
}
