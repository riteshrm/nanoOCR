import torch
import torch.nn as nn

class SqueezeExcitationModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()

        reduced_channels = max(1, channel // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, reduced_channels, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channel, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
    
class DWSEPWConv(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.dw_conv = nn.Sequential(nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish())
        self.squeeze_excitation = SqueezeExcitationModule(in_channels)
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dw_conv(hidden_states)
        hidden_states = self.squeeze_excitation(hidden_states)
        hidden_states = self.pw_conv(hidden_states)
        return hidden_states
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.5, conv_type='norm'):
        super().__init__()
        if conv_type=='dwsepw':
            self.cnn = DWSEPWConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.maxpool = nn.MaxPool2d(pool_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_classes, height=32, conv_type='norm'):
        super().__init__()
        channels = [ 1, 64, 128, 256, 256]
        pooling = [(2, 2), (2, 2), (2, 1), (2, 1)]
        self.cnns = nn.ModuleList([CNNBlock(in_channels=channels[i], out_channels=channels[i+1], pool_size=pool_size, conv_type=conv_type) for i, pool_size in enumerate(pooling)])
        self.rnn = nn.LSTM(256 * (height // 16), 128, num_layers=2,
                           bidirectional=True, dropout=0.4, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
 
    def forward(self, x):
        for cnnblock in self.cnns:
            x = cnnblock(x)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * h)   # (B, T, C*h)
        
        x, _ = self.rnn(x)
        return self.fc(x).permute(1, 0, 2)


class CRNN2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.act_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))


        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.act_3 = nn.ReLU()

        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.act_4 = nn.ReLU()
        self.pool_4 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.act_5 = nn.ReLU()
        self.batch_norm_5 = nn.BatchNorm2d(256)
        
        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.act_6 = nn.ReLU()
        self.batch_norm_6 = nn.BatchNorm2d(256)
        self.pool_6 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv_7 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=2)
        self.act_7 = nn.ReLU()
        
        self.blstm_1 = nn.LSTM(input_size=64,hidden_size=128,num_layers=2,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.act_3(x)

        x = self.conv_4(x)
        x = self.act_4(x)
        x = self.pool_4(x)

        x = self.conv_5(x)
        x = self.act_5(x)
        x = self.batch_norm_5(x)

        x = self.conv_6(x)
        x = self.act_6(x)
        x = self.batch_norm_6(x)
        x = self.pool_6(x)

        x = self.conv_7(x)
        x = self.act_7(x)

        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * h)   # (B, T, C*h)
        x, _ = self.blstm_1(x)
        return self.fc(x).permute(1, 0, 2)