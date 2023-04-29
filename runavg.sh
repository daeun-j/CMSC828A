cd data/utils
python3 run.py -d tiny_imagenet -a 0.1 -cn 100
cd ../../

# # run FedAvg under default setting.

cd ./src/server

python3 fedavg.py -d tiny_imagenet -m res18 -ge 100 -le 2 -jr 0.5 -lr 0.1 -mom 0.4 -wd 0.0 -bs 32 
python3 fedavg.py -d tiny_imagenet -m res18 -ge 50 -le 4  -jr 0.5 -lr 0.1 -mom 0.4 -wd 0.0 -bs 32 
python3 fedavg.py -d tiny_imagenet -m res18 -ge 10 -le 10 -jr 0.5 -lr 0.1 -mom 0.4 -wd 0.0 -bs 32 
python3 fedavg.py -d tiny_imagenet -m res18 -ge 5 -le 40 -jr 0.5 -lr 0.1 -mom 0.4 -wd 0.0 -bs 32 
python3 fedavg.py -d tiny_imagenet -m res18 -ge 2 -le 100 -jr 0.5 -lr 0.1 -mom 0.4 -wd 0.0 -bs 32 


