import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,
                 kernel_sizes=[5, 5, 3, 3],
                 n_filters=[64, 64, 128, 128],
                 num_channels=12,
                 num_hidden=20,
                 num_labels=3):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=n_filters[0], kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=n_filters[0], out_channels=n_filters[1], kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=n_filters[1], out_channels=n_filters[2], kernel_size=kernel_sizes[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # added
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=n_filters[2], out_channels=n_filters[3], kernel_size=kernel_sizes[3]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )


        self.fc0 = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(32384, num_hidden),
            # nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            # nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x)
        return (x)

if __name__ == '__main__':
    x = torch.rand(64, 12, 4096)
    mdl = CNN()
    y = mdl(x)
    print('all good')