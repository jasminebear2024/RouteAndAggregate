ssh -p 996 ubuntu@10.112.190.137

123456

cd ../deeplearning/miniconda3/bin

source activate lwc_torch_37

cd ../../../deeplearning/lwc/pycharm_remote/routing





[]

curl 'http://10.3.8.211/login' --data 'user=2020010046&pass=1022mnMN'





cd anaconda3/envs/lwc_FL

source activate lwc_FL

cd ../../../liweicai/lwc_d2dFL

cd liweicai/lwc_d2dFL



windows

echo ". /c/users/11209/miniconda3/etc/profile.d/conda.sh" >> ~/.profile

source activate fedml

cd ../../../../

 cd /d/Cloud/NutStorage/KittyFox/python/FedML_v3/lwc_d2dFL/





echo ". /c/programdata/miniconda3/etc/profile.d/conda.sh" >> ~/.profile

source activate lwcFL



nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 cifar10 ./data/cifar10 resnet56 hetero 200 1 0.03 sgd 0 5 10 2 0 1 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &



nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 shakespeare ./data/shakespeare/datasets rnn hetero 100 1 0.8 sgd 0 5 10 2 0 1 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &

nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 fed_shakespeare  ./data/fed_shakespeare/datasets rnn  hetero 1000  1 0.8 sgd 0 5 10 2 0 1 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &

sh run_d2d_fedavg_standalone_pytorch.sh 0 20 fed_shakespeare  ./data/fed_shakespeare/datasets rnn  hetero 1000  1 0.8 sgd 0 5 10 0 0 1 fed_cifar100_train.h5

0 0

2 0 

3 1

3 5

5 1

5 5

6 0

8 0

12,500  25,000  50,000  100,000



0.5 0.7 0.9
