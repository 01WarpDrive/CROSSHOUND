# Host Detector

## Usage

### Quick Demo
```bash
python eval.py --dataset [DATASET]
```

### Training from Scratch

Preprocess data and construct provenance graphs:
```bash
cd ./utils/
python parser.py --dataset [DATASET]
```

Start graph representation learning:
```bash
python train.py --dataset [DATASET]
```

Anomaly detection:
```bash
python eval.py --dataset [DATASET]
```
