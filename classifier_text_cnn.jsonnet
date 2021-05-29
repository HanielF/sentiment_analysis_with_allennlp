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
            "ngram_filter_sizes":[2, 3, 4]
        },
        "dropout": 0.5,
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
        },
        "num_epochs": 50,
        "cuda_device": 0
    }
}
