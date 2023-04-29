# cd data/utils
# python3 run.py -d tiny_imagenet -a 0.1 -cn 100
# cd ../../

# # run FedAvg under default setting.

cd ./src/server


python3 fedprox.py -d tiny_imagenet -m res18 -ge 100 -le 10 -prox_lambda 2 -jr 0.7 -lr 0.02
python3 fedprox.py -d tiny_imagenet -m res18 -ge 100 -le 10 -prox_lambda 1 -jr 0.7 -lr 0.02
python3 fedprox.py -d tiny_imagenet -m res18 -ge 100 -le 10 -prox_lambda 0 -jr 0.7 -lr 0.02

python3 fedavg.py -d tiny_imagenet -m res18 -ge 100 -le 10 -jr 0.7 -lr 0.02

 
