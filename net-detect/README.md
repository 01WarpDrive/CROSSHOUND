# Net Detector

## Usage
* Only need to train FastText and VAE models once. 

Train the word embedding model FastText:
```bash
python train_fasttext.py
```

Preprocess data:
```bash
python preprocess.py --dataset [DATASET]
```

Train VAE model:
```bash
python train_vae.py
```

Anomaly detection:
```bash
python eval.py --dataset [DATASET]
```



## Quick eval

When the training is completed, conduct the entire process of detection and testing quickly：

```
python stream_process.py --dataset [DATASET]
```

