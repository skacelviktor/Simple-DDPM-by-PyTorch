import torch.nn as nn
import torch
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self, dim, device):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.Silu = nn.SiLU()
        self.lin2 = nn.Linear(dim * 4, dim)
        self.inv_freq = 10000 ** (torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=device) / (self.dim // 2))

    def forward(self, time_steps):
        
        # timesteps B -> B, 1 -> B, temb_dim
        t_emb = time_steps[:, None].repeat(1, self.dim // 2) / self.inv_freq
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

        t_emb = self.lin1(t_emb)
        t_emb = self.Silu(t_emb)
        t_emb = self.lin2(t_emb)

        return t_emb
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                   nn.SiLU())
        
        self.time_mlp = nn.Sequential(nn.SiLU(),
                                      nn.Linear(time_emb_dim, out_ch))
        
        self.conv2 = nn.Sequential(nn.SiLU(),
                                   nn.Conv2d(out_ch, out_ch, 3, padding=1))
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        # Přidáme časovou informaci (broadcast přes H a W)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        return h + self.shortcut(x)
    
class Down_block(nn.Module):
    def __init__(self, in_channles, out_channels, t_dim):
        super(Down_block, self).__init__()

        self.Res_block = ResBlock(in_channles, out_channels, t_dim)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        skip = self.Res_block(x, t)

        x = self.pool(skip)

        return x, skip
    
class Up_block(nn.Module):
    def __init__(self, in_channles, out_channels, t_dim):
        super(Up_block, self).__init__()

        self.up = nn.ConvTranspose2d(in_channles, in_channles, kernel_size=2, stride=2)

        self.Res_block = ResBlock(in_channles*2, out_channels, t_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        #add skip from encoder
        x = torch.cat([x, skip], dim=1)
        x = self.Res_block(x, t)

        return x

class U_net(nn.Module):
    def __init__(self, device, in_channels = 1, t_dim = 256):
        super(U_net, self).__init__()

        #input (28, 28, 1) and time

        #time embeding
        self.time_mlp = TimeEmbedding(t_dim, device)

        #encoder
        self.Down_blocks = nn.ModuleList([Down_block(in_channels, 64, t_dim), #(14, 14, 64)
                            Down_block(64, 128, t_dim)]) #(7, 7, 128)
        
        #bottleneck
        self.bottleneck = ResBlock(128, 128, t_dim)

        #decoder
        self.Up_blocks = nn.ModuleList([Up_block(128, 64, t_dim), #(14, 14, 64)
                          Up_block(64, 64, t_dim)]) #(28, 28, 64)
        
        #output layer
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, t):
        #time embeding
        t_emb = self.time_mlp(t)

        #Down blocks
        skips = []
        for down_block in self.Down_blocks:
            x, skip = down_block(x, t_emb)

            skips.append(skip)

        #bottleneck
        x = self.bottleneck(x, t_emb)

        #Up blocks
        for idx, skip in enumerate(reversed(skips)):
            x = self.Up_blocks[idx](x, skip, t_emb)

        x = self.final_conv(x)

        return x