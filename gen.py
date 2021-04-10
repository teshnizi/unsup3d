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

# =================================================================================
# =================================================================================
# =================================================================================

is_train = False 
is_test = not is_train

trainer.model.to_device(trainer.device)
trainer.current_epoch = trainer.load_checkpoint(optim=False)

with torch.no_grad():
    trainer.model.set_eval()
    # for iter, image in enumerate(trainer.test_loader):
        
        # input_im = image.to(trainer.model.device) *2.-1.
        # b, c, h, w = input_im.shape

        # x = input_im

        # print(f'Input shape: {x.shape}\n------------------------')
        # for l in list(trainer.model.netA.network.children())[:-2]:
        #     x = l(x)
        #     print(l)
        #     print(x.shape)
        #     print('------------------------')

        # feature_maps = x.cpu()
        # print(feature_maps.shape)
        # square = 8
        # ix = 1
        # for _ in range(square):
        #     for _ in range(square):
        #         # specify subplot and turn of axis
        #         ax = pyplot.subplot(square, square, ix)
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         # plot filter channel in grayscale
        #         pyplot.imshow(feature_maps[0, ix-1, :, :])
        #         ix += 1
        # # show the figure
        # pyplot.savefig('albedo.png')


        # # canon_albedo = trainer.model.netA(input_im)  # Bx3xHxW
        # # print(canon_albedo.shape)


