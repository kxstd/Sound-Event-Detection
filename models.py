import torch
import torch.nn as nn

class Pool(nn.Module):
    def __init__(self, pool_method, output_dim):
        super().__init__()
        self.method = pool_method
        self.alpha = nn.Parameter(torch.full((output_dim, ), 1.0))
    def forward(self, logits, decision):
        if self.method == "linear_soft":
            return (decision**2).sum(1) / decision.sum(1)
        if self.method == "max":
            return torch.max(decision, dim=1)[0]
        if self.method == "mean":
            return torch.mean(decision, dim=1)
        if self.method == "soft":
            w = torch.softmax(decision, dim=1)
            return torch.sum(decision * w, dim=1)
        raise


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)

class Crnn(nn.Module):
    def __init__(self, num_freq, num_class, pool_method):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      num_freq)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = Pool(pool_method,num_class)
        self.outputlayer = nn.Linear(256, num_class)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x,decision_time).clamp(1e-7, 1.).squeeze(1)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        
        return {"frame_prob":decision_time, "clip_prob":decision}