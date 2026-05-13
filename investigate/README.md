# Attack Investigation

## Usage
* For convenience, the matching steps of OpTC have been completed in the network detection module. 

Match host&net data (to test the overhead):
```bash
python match.py --dataset [DATASET]
```

Reconstruct attack paths:
```bash
python stp.py --dataset [DATASET]
```
