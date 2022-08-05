## File Structure:

Main codes
  - SFAT.py
  - Centralized_AT.py
  - update.py
  - utils.py 
  - attack_generator.py

Data split and model structure
  - sampling.py
  - models.py

Other utils and parameter setups
  - options.py
  - logger.py
  - eval_pgd.py

## Environment

  - Python (3.8)
  - Pytorch (1.7.0 or above)
  - torchvision
  - CUDA
  - Numpy

## Training example

``` 
CUDA_VISIBLE_DEVICES='0' python SFAT.py --dataset=cifar-10 --local_ep=10 --local_bs=32 --iid=0 --epochs=100 --num_users=5 --agg-opt='FedAvg' --agg-center='FAT' --out-dir='../output_results_FAT_FedAvg'
```

``` 
CUDA_VISIBLE_DEVICES='0' python SFAT.py --dataset=cifar-10 --local_ep=10 --local_bs=32 --iid=0 --epochs=100 --num_users=5 --agg-opt='FedAvg' --agg-center='SFAT' --pri=1.2 --out-dir='../output_results_SFAT_FedAvg'
```
