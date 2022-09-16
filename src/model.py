import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType as TT  # type: ignore
from torchtyping import patch_typeguard   # type: ignore
patch_typeguard()
from typeguard import typechecked  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride,
		 dropoutRate=0.0, activate_before_residual=False, momentum=0.001):
        super(ResidualBlock, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dropoutRate = dropoutRate
        self.activate_before_residual = activate_before_residual
        self.identity = nn.Identity()
        if stride >= 2 or inChannels != outChannels:
            # 1x1 convolution to change the number of output layers.
            self.identity = nn.Conv2d(inChannels, outChannels,
                                      1, stride, bias=False)

        
        # realistic-ssl-evaluation-pytorch works with default momentum
        # (0.1)
        self.bn1 = nn.BatchNorm2d(inChannels, momentum=momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3,
			       stride=stride, padding=1, bias=False)

        # realistic-ssl-evaluation-pytorch works with default momentum
        # (0.1)
        self.bn2 = nn.BatchNorm2d(outChannels, momentum=momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(outChannels, outChannels,
			       kernel_size=3, stride=1, padding=1, bias=False)

        self.training = True
        
    @typechecked
    def forward(self, x: TT[-1, "inChannels", -1, -1]) -> TT[-1, "outChannels", -1, -1]:
        if self.activate_before_residual == True:
            x = self.bn1(x)
            x = self.relu1(x)
            out = x
        else:
            out = self.bn1(x)
            out = self.relu1(out)
        out = self.conv1(out)
        out = F.dropout(out, p=self.dropoutRate, training=self.training)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.add(self.identity(x), out)
        return out

class ResidualNetwork(nn.Module):
    def __init__(self, numBlocks, inChannels, outChannels, stride,
                 dropoutRate=0.0, momentum=0.001, activate_before_residual=False):
        super(ResidualNetwork, self).__init__()
        #self.blocks = []
        blocks = []
        for l in range(int(numBlocks)):
            blocks += [ResidualBlock(
                l == 0 and inChannels or outChannels,
                outChannels,
                l == 0 and stride or 1,
                dropoutRate,
                activate_before_residual =
                activate_before_residual and l == 0, momentum=momentum)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
        # for block in self.blocks:
        #     out = block(x)
        #     x = out
        # return out

class WideResNet(nn.Module):
    def __init__(self, numClasses, depth=28, widen_factor=2, dropoutRate=0.0,
                 momentum=0.001
                 ):
        super(WideResNet, self).__init__()
        assert((depth - 4) % 6 == 0)
        assert depth >= 10
        # assert dropoutRate == 0.0, "Not yet implemented"
        numChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        numBlocks = (depth - 4) / 6
        self.nChannels = numChannels[3]
        self.momentum = momentum # nn.BatchNorm2d(1).momentum  # default momentum

        # 1st conv before any network unit
        self.init_conv = nn.Conv2d(3, numChannels[0], kernel_size=3,
				   stride=1, padding=1, bias=False)
        # 1st unit
        self.unit1 = ResidualNetwork(numBlocks, numChannels[0],
				     numChannels[1], 1, dropoutRate, momentum,
				     activate_before_residual=True)
        # 2nd unit
        self.unit2 = ResidualNetwork(numBlocks, numChannels[1],
				     numChannels[2], 2, dropoutRate, momentum)
        # 3rd unit
        self.unit3 = ResidualNetwork(numBlocks, numChannels[2],
				     numChannels[3], 2, dropoutRate, momentum)

        # global average pooling and classifier
        # bn has default momentum (0.1).
        self.bn = nn.BatchNorm2d(numChannels[3], momentum=momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(numChannels[3], numClasses)

        self._init_params()

        self.device = DEVICE
        self.to(self.device)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @typechecked
    def forward(self, x: TT[-1, 3, 32, 32]) -> TT[-1, 10]:
        # breakpoint()
        x = x.to(self.device)
        # Comments for widen_factor=2.
        out = self.init_conv(x) # 16, 32, 32
        out = self.unit1(out) # 32, 32, 32
        out = self.unit2(out) # 64, 16, 16
        out = self.unit3(out) # 128, 8, 8

        # 4th unit is implemented here:
        out = self.bn(out)
        out = self.relu(out)

        # Make 'out' (which for widen_factor=2 is B x 128 x 8 x 8) 8
        # times smaller.
        out = self.pool(out).squeeze() 

        # Should equivalent to squeeze, but less safe (can it change
        # the batch dim?)
        # out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out

    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        import pathlib

        # Save e.g. 'epoch_017.pt' in the save dir
        save_dir_p = pathlib.Path(save_dir) / f'epoch_{epoch:03d}.pt'

        torch.save(self.state_dict(), str(save_dir_p))

    # Adapted from
    # https://github.com/aleloi/ast-classification/blob/main/utils.py
    def load_latest_checkpoint(self, save_dir: str) -> int:
        import pathlib
        import re

        p = pathlib.Path(save_dir)

        # all files ending with '.pt'
        files_p = [x for x in p.iterdir() if x.parts[-1].endswith(".pt")]
        if not files_p:
            assert False, ("Can't resume if there are no checkpoints. "
                           "There must be at least one completed epoch "
                           "to resume from.")

        # Get the one that changed last.
        latest_epoch_file = max(files_p, key=lambda x: x.stat().st_mtime)

        # Extract the epoch:
        m = re.match("epoch_(?P<epoch>\d+).pt",
            latest_epoch_file.parts[-1])
        if m is None:
            assert False
        epoch = int(m.group('epoch'))

        print(f"Path {p} contains {len(files_p)} checkpoints.\n"
              f"Loading {latest_epoch_file.parts[-1]} out of all checkpoints")

        # Handle cpu/gpu tensors
        kwargs = {}
        if torch.cuda.device_count() == 0:
            kwargs['map_location'] = torch.device('cpu')

        # log if smth went wrong
        # print(f"Expected keys: {self.state_dict().keys()}")
        miss_unexp = self.load_state_dict(torch.load(latest_epoch_file, **kwargs))
        print(f"Missing/unexpected: {miss_unexp}")

        # Put back tensors to where they are expected.
        self.to(self.device)

        return epoch

    def set_training(self, training: bool):
        if training:
            self.train()
        else:
            self.eval()
