## 记录模型训练和验证的结果

### TextCNN

```json
{
  "best_epoch": 0,
  "peak_worker_0_memory_MB": 3181.39453125,
  "peak_gpu_0_memory_MB": 98.9375,
  "training_duration": "0:11:44.806978",
  "training_start_epoch": 0,
  "training_epochs": 49,
  "epoch": 49,
  "training_accuracy": 0.965567417482071,
  "training_loss": 0.06314526099391614,
  "training_worker_0_memory_MB": 3181.39453125,
  "training_gpu_0_memory_MB": 98.9375,
  "validation_accuracy": 0.768348623853211,
  "validation_loss": 6.002619594335556,
  "best_validation_accuracy": 0.8073394495412844,
  "best_validation_loss": 0.42884890522275654
}
```

### BiLSTM

```json
{
  "best_epoch": 5,
  "peak_worker_0_memory_MB": 3100.875,
  "peak_worker_1_memory_MB": 3067.96484375,
  "peak_gpu_0_memory_MB": 108.94921875,
  "peak_gpu_1_memory_MB": 108.86376953125,
  "training_duration": "0:29:00.362737",
  "training_start_epoch": 0,
  "training_epochs": 99,
  "epoch": 99,
  "training_accuracy": 0.9831177894252328,
  "training_loss": 0.05053796984793389,
  "training_worker_0_memory_MB": 3100.875,
  "training_worker_1_memory_MB": 3067.96484375,
  "training_gpu_0_memory_MB": 108.94921875,
  "training_gpu_1_memory_MB": 108.86376953125,
  "validation_accuracy": 0.7522935779816514,
  "validation_loss": 3.1775100316320146,
  "best_validation_accuracy": 0.8291284403669725,
  "best_validation_loss": 0.46113214216062
}
```

### BERT Base with pooling

```json
{
  "best_epoch": 0,
  "peak_worker_0_memory_MB": 4152.94140625,
  "peak_gpu_0_memory_MB": 0,
  "training_duration": "1:32:29.037002",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_accuracy": 0.9940756358669023,
  "training_loss": 0.01622601636273874,
  "training_worker_0_memory_MB": 4152.94140625,
  "training_gpu_0_memory_MB": 0.0,
  "validation_accuracy": 0.908256880733945,
  "validation_loss": 0.5078080969729403,
  "best_validation_accuracy": 0.9323394495412844,
  "best_validation_loss": 0.19536435732259116
}
```

### BERT Large with pooling

```json
{
  "best_epoch": 2,
  "peak_worker_0_memory_MB": 5080.01171875,
  "peak_gpu_0_memory_MB": 0,
  "training_duration": "3:53:35.280571",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_accuracy": 0.9929174894950185,
  "training_loss": 0.02284079474960264,
  "training_worker_0_memory_MB": 5080.01171875,
  "training_gpu_0_memory_MB": 0.0,
  "validation_accuracy": 0.926605504587156,
  "validation_loss": 0.34686793256850074,
  "best_validation_accuracy": 0.9311926605504587,
  "best_validation_loss": 0.19148897620798047
}
```

### BERT base with LSTM

```json
{
  "best_epoch": 0,
  "peak_worker_0_memory_MB": 4178.55859375,
  "peak_gpu_0_memory_MB": 1830.3349609375,
  "training_duration": "0:53:20.329597",
  "training_start_epoch": 0,
  "training_epochs": 4,
  "epoch": 4,
  "training_accuracy": 0.9858052829292194,
  "training_loss": 0.04194650748686771,
  "training_worker_0_memory_MB": 4178.55859375,
  "training_gpu_0_memory_MB": 1830.3349609375,
  "validation_accuracy": 0.9151376146788991,
  "validation_loss": 0.34153107094000107,
  "best_validation_accuracy": 0.9243119266055045,
  "best_validation_loss": 0.2112232671298702
}
```

### RoBERTa with pooling

```json
{
  "best_epoch": 1,
  "peak_worker_0_memory_MB": 6039.23046875,
  "peak_gpu_0_memory_MB": 5639.6064453125,
  "training_duration": "4:03:17.076677",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_accuracy": 0.9777724984780769,
  "training_loss": 0.07260703351776329,
  "training_worker_0_memory_MB": 6039.23046875,
  "training_gpu_0_memory_MB": 5639.6064453125,
  "validation_accuracy": 0.9334862385321101,
  "validation_loss": 0.2241931054484817,
  "best_validation_accuracy": 0.9564220183486238,
  "best_validation_loss": 0.13505156283531713
}
```
