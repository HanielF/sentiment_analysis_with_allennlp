local cur_model_name = "roberta-large";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": cur_model_name,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": cur_model_name,
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/SST-2/train.tsv",
    "validation_data_path": "data/SST-2/dev.tsv",
    "model": {
        "type": "bert_pool_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": cur_model_name
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": cur_model_name,
            "requires_grad": true
        }
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
        "num_epochs": 10
    },
    "distributed": {
        "cuda_devices": [0, 1],
    }
}

