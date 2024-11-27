Topology type: 

0 -- R\&A w/ model subst.

2 -- R\&A w/ coeff. norm.

3 -- AaYG w/ model subst.

5 -- AaYG w/ coeff. norm.

6 -- C-FL w/ model subst.

8 -- C-FL w/ coeff. norm.



Dataset Example: 

nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 shakespeare ./data/shakespeare/datasets rnn hetero 100 1 0.8 sgd 0 5 10 2 0 1 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &

nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 cifar10 ./data/cifar10 resnet56 hetero 200 1 0.03 sgd 0 5 10 2 0 10 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &

nohup sh run_d2d_fedavg_standalone_pytorch.sh 0 20 fed_shakespeare ./data/fed_shakespeare/datasets rnn hetero 1000 1 0.8 sgd 0 5 10 3 5 1 fed_cifar100_train.h5 > ./fedavg_standalone.txt 2>&1 &
