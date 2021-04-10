import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3D


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, Unsup3D)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

data_loader = trainer.test_loader
for x in data_loader:
    print(x.shape)

with torch.no_grad():
    m = trainer.run_epoch(trainer.test_loader, is_test=True)
