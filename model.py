import torch
import torch.nn as nn

class ZeroDCE(nn.Module):
    def __init__(self, channels=3, iterations=8):  #iterations = how may times to apply the enhancement curve 
        super(ZeroDCE, self).__init__() #initialise the base class nn.Module
        self.iterations = iterations
        
        self.relu = nn.ReLU(inplace=True) #introduces non-linearity

        # Feature extraction layers
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32 * 2, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32 * 2, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, channels * iterations, 3, 1, 1)

        self.tanh = nn.Tanh() #squeezes the final output values between -1 & 1

    def forward(self, x):
        #extracts features form 4 layers
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        #fuses features from different layers
        x5 = torch.cat([x3, x4], 1)
        x5 = self.relu(self.conv5(x5))
        x6 = torch.cat([x2, x5], 1)
        x6 = self.relu(self.conv6(x6))

        #predict enhancement curves : estimate 8 3-channel curves
        x_r = self.tanh(self.conv7(x6))
        r = torch.split(x_r, 3, dim=1)

        # Apply curves iteratively
        x_enhanced = x
        for i in range(self.iterations):
            x_enhanced = x_enhanced + r[i] * (torch.pow(x_enhanced, 2) - x_enhanced)
            x_enhanced = torch.clamp(x_enhanced, 0, 1)  #ensures the pixel values stay b/w 0 & 1

        return x_enhanced, r
