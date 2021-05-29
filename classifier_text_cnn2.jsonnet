{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "spacy"
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "data/SST-2/train.tsv",
    "validation_data_path": "data/SST-2/dev.tsv",
    "model": {
        "type": "text_cnn_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256,
                    "trainable":true
                }
            }
        },
        "encoder": {
            "type": "cnn",
            "embedding_dim": 256,
            "num_filters": 100,
            "ngram_filter_sizes":[3, 4, 5]
        },
        "dropout": 0.5,
        "regularizer":{
            "regexes": [["weight","l2"]]
        }
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "eps":1e-07,
            "lr": 0.001
        },
        "num_epochs": 50,
        "cuda_device": 0
    }
}
