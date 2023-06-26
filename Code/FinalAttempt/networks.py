import torch

# importing the utilities module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utilities as utils

class InitGuessNetwork(torch.nn.Module):
    def __init__(self):
        super(InitGuessNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode='reflect')
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False, padding_mode='reflect')
        self.conv4 = torch.nn.Conv2d(1, 1, kernel_size=9, padding=4, bias=False, padding_mode='reflect')

        # Map to store the solved d_const matrices
        self.dConstMatrixMap = {}

    def forward(self, array, bounds, d_const):
        # Clamp with domain bounds
        x = array.clamp(min=bounds[0][0], max=bounds[0][1])

        # Subtract effect of d_const at the beginning
        N = array.shape[2]
        dConstMatrix = self.getDConstMatrix(d_const, N)
        x = x - dConstMatrix

        # For normalization
        x_min = x.min()
        #x_range = x.max() - x_min
        #x_mean = x.mean()

        # Run the convolutions
        x = x - x_min
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x + x_min

        # Extract the boundaries
        x = torch.cat((x[:,0,:], x[:,:,-1], x[:,-1,:], x[:,:,0])).unsqueeze(0)

        # Clamp with boundary bounds
        x = x.clamp(min=bounds[0][2], max=bounds[0][3])

        return x

    def getDConstMatrix(self, d_const, N):
        # Just memorizing to avoid recalculation every time
        ret = self.dConstMatrixMap.get(N)
        if ret is None:
            ret = utils.solvePDE(
                torch.zeros((1, 4, N)),
                torch.tensor(-10),
                0
            )[:, 1:-1, 1:-1]
            self.dConstMatrixMap[N] = ret
        return ret * (d_const.item() / -10)

class ConvLSTMLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(ConvLSTMLayer, self).__init__()

        self.reset()

        # 2D convolutional layer
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def reset(self):
        self.hidden = None

    def forward(self, x):
        # Apply the 2D convolutional layer to the input sequence
        x = self.conv2d(x)
        x = torch.nn.functional.relu(x)
        x = x.permute(0, 2, 3, 1)  # permute to (batch_size, height, width, channels)

        # Flatten the output of the convolutional layer
        batch_size, height, width, channels = x.size()
        x = x.view(batch_size, height * width, channels)

        # Apply the LSTM layer to the output of the convolutional layer
        output, self.hidden = self.lstm(x, self.hidden)

        # Reshape the output of the LSTM layer to match the input size
        output = output.view(batch_size, height, width, -1)
        output = output.permute(0, 3, 1, 2)  # permute back to (batch_size, channels, height, width)

        return output

class OptimizerNetwork(torch.nn.Module):
    def __init__(self):
        super(OptimizerNetwork, self).__init__()

        self.firstRun = True

        # Learning rate and beta values
        self.betas = torch.nn.Linear(0, 3)
        self.betas.bias.data = torch.tensor([0.9, 0.999, 0.9])
        self.lrs = torch.nn.Linear(0, 3)
        self.lrs.bias.data = torch.tensor([0.001, 0.001, 0.001])

        # Neural network that has both convolution and lstm
        self.relu = torch.nn.ReLU()
        self.tempConv1 = ConvLSTMLayer(3, 8, 4)
        self.tempConv2 = ConvLSTMLayer(4, 8, 4)
        self.conv1 = torch.nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect')

        #self.stepNumber = 0
        #self.first_net_delta = None

    def reset(self):
        self.firstRun = True

        self.tempConv1.reset()
        self.tempConv2.reset()

        #self.stepNumber = 0
        #self.first_net_delta = None

    def forward(self, x, grad, eps=1e-08):
        if self.firstRun:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
            self.rmsPropCache = torch.zeros_like(grad)
            self.firstRun = False

        # Get learning rates and betas
        betas = self.betas(torch.tensor([]))
        lrs = self.lrs(torch.tensor([]))

        # Using Adam
        self.m = betas[0] * self.m + (1 - betas[0]) * grad
        self.v = betas[1] * self.v + (1 - betas[1]) * (grad**2)
        m_hat = self.m / (1 - betas[0])
        v_hat = self.v / (1 - betas[1])
        adam_delta = m_hat / (torch.sqrt(v_hat) + eps)

        # Using RMSProp
        self.rmsPropCache = betas[2] * self.rmsPropCache + (1 - betas[2]) * (grad ** 2)
        rmsProp_delta = grad / (torch.sqrt(self.rmsPropCache) + eps)

        # Using a neural network that has both convolution and lstm
        h = torch.cat((grad, grad**2, grad**3))
        h = h.unsqueeze(0)
        h = torch.nn.functional.relu(self.tempConv1(h))
        h = torch.nn.functional.relu(self.tempConv2(h))
        h = torch.nn.functional.relu(self.conv1(h))
        h = h.squeeze(0)
        # Consider the sign of the gradients and x; relu removes it all.
        grad_sign = torch.sign(grad)
        net_delta = grad_sign * h

        # Update x
        x = x - lrs[0] * adam_delta - lrs[1] * rmsProp_delta - lrs[2] * net_delta

        '''
        if self.first_net_delta == None:
            self.first_net_delta = net_delta
        diff_net_delta = net_delta - self.first_net_delta
        self.csvFile.write(
            f'{self.stepNumber},' +
            f'{x.shape[2]},' +
            f'{(lrs[0] * adam_delta).abs().mean().item()},' +
            f'{(lrs[1] * rmsProp_delta).abs().mean().item()},' +
            f'{(lrs[2] * net_delta).abs().mean().item()},' +
            f'{(lrs[2] * diff_net_delta).abs().mean().item()}\n'
        )
        self.csvFile.flush()
        self.stepNumber += 1
        '''

        return x
