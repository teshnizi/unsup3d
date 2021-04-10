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

trainer.model.to_device(trainer.device)
trainer.current_epoch = trainer.load_checkpoint(optim=False)
if trainer.test_result_dir is None:
    trainer.test_result_dir = os.path.join(trainer.checkpoint_dir, f'test_results_{trainer.checkpoint_name}'.replace('.pth',''))
print(f"Saving testing results to {trainer.test_result_dir}")

# =================================================================================
# =================================================================================
# =================================================================================


is_train = False 
is_test = not is_train

with torch.no_grad():
    trainer.model.set_eval()
    for iter, x in enumerate(trainer.test_loader):
        # print(f'X: {x.shape}')
        # m = trainer.model.forward(x)
        # print(m)
        # if is_train:
        #     trainer.model.backward()
    
        input_im = x.to(trainer.model.device) *2.-1.
        b, c, h, w = input_im.shape
        canon_albedo = trainer.model.netA(input_im)  # Bx3xHxW
        print(canon_albedo.shape)

# data_loader = trainer.test_loader
# model = trainer.model
# netD = model.netD
# netA = model.netA
# netL = model.netL
# netV = model.netV
# netC = model.netC

# print(netA)
    