# SFAT: Preventing Intensified Heterogeneity for Adversarially Robust Decentralized Models

This is the source code of our proposed framework ```Slack Federated Adversarial Training (SFAT)```.

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

## Quick overview of our SFAT

Following the conventional federated learning realization, we realizes the overall framework of ```SFAT``` in ```SFAT.py``` which coordinate the local optimization part in ```update.py``` and the aggregation functions in ```utils.py```.

In ```SFAT.py```, we get the local model in each client and aggregate the global model.

~~~python
# local updates
for idx in idxs_users:
    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, alg=args.agg_opt, anchor=global_model, anchor_mu=args.mu, local_rank=ipx, method=args.train_method)
            ''' ''' 
# aggregation method
if args.agg_center == 'FedAvg':
    global_weights = average_weights(local_weights)
if args.agg_center == 'SFAT':
    ''' '''
    global_weights = average_weights_alpha(local_weights, idt, idtxnum, args.pri)
~~~

In ```updates.py```, we realize the local training on each client for adversarial training and defined the ```LocalUpdate()```.

In ```utils.py```, we realize the aggregation methods and define the FAT, i.e., ```average_weights(local_weights)``` and SFAT ```average_weights_alpha``` as well as their unequal versions.

### Data split 

We realize the operation of data split in ```sampling.py``` and utilized in ```utils.py``` for generate local data loader for each client.

~~~python
def get dataset(args):
    ''' ''' 
    user_groups = cifar_noniid_skew(train_dataset, args.num_users)
    ''' '''
    return train_dataset, test_dataset, user_groups
~~~

### Choosing different optimization and aggregation methods

To choose different federated optimization methods (e.g., FedAvg, FedProx, Scaffold) and the aggregations (e.g., FAT and SFAT) for training robust federated model. We can used defined parameter in our ```options.py```:

~~~python
parser.add_argument('--agg-opt',type=str,default='FedAvg',help='option of on-device learning: FedAvg, FedProx, Scaffold')
parser.add_argument('--agg-center',type=str,default='FedAvg',help='option of aggregation: FedAvg, SFAT')
~~~

### Running example

To train federated robust model, we provide examples below to use our code:

~~~bash
CUDA_VISIBLE_DEVICES='0' python SFAT.py --dataset=cifar-10 --local_ep=10 --local_bs=32 --iid=0 --epochs=100 --num_users=5 --agg-opt='FedAvg' --agg-center='FAT' --out-dir='../output_results_FAT_FedAvg'
~~~

~~~bash
CUDA_VISIBLE_DEVICES='0' python SFAT.py --dataset=cifar-10 --local_ep=10 --local_bs=32 --iid=0 --epochs=100 --num_users=5 --agg-opt='FedAvg' --agg-center='SFAT' --pri=1.2 --out-dir='../output_results_SFAT_FedAvg'
~~~

### Evaluation

To evaluate our trained model using various attack methods, we provide the ```eval_pgd.py``` contains different evaluation metrics for natural and robust performance. You can run the following script with your model path to conduct evaluation:

~~~bash
CUDA_VISIBLE_DEVICES='0' python eval_pgd.py --net [NETWORK STRUCTURE] --dataset [DATASET] --model_path [MODLE PATH]
~~~

Actually, during the training, we also provide the accuracy track via ```logger.py``` to save the model performance in each epoch.

### To extend the yourself method in our framework

Either the local optimization or aggregation method can be re-designed based on our framework in the corresponding ```updates.py``` and ```utils.py``` part. 


