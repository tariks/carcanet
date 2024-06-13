import torch
import numpy as np
import torch.nn as nn
from keras.layers import ReLU
from baseline_aug import get_unet


def my_unet(do=0, activation=nn.ReLU):
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )
            self.pool3 = nn.MaxPool2d(2, 2)

            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )
            self.pool4 = nn.MaxPool2d(2, 2)

            self.conv5 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )

            self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )

            self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv7 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )

            self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv8 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )

            self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv9 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Dropout(do),
                activation(),
            )

            self.conv10 = nn.Conv2d(32, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.conv1(x)
            x = self.pool1(x1)

            x2 = self.conv2(x)
            x = self.pool2(x2)

            x3 = self.conv3(x)
            x = self.pool3(x3)

            x4 = self.conv4(x)
            x = self.pool4(x4)

            x = self.conv5(x)

            x = torch.cat((self.up6(x), x4), dim=1)
            x = self.conv6(x)

            x = torch.cat((self.up7(x), x3), dim=1)
            x = self.conv7(x)

            x = torch.cat((self.up8(x), x2), dim=1)
            x = self.conv8(x)

            x = torch.cat((self.up9(x), x1), dim=1)
            x = self.conv9(x)

            x = self.conv10(x)
            x = self.sigmoid(x)

            return x

    model = UNet()
    criterion = nn.MSELoss()
    # TODO style loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # lr should be smaller than original of 1e-3
    return model, criterion, optimizer


# Load the Keras model weights
file_path = "baseline_unet_aug_do_0.1_activation_ReLU_weights.best.hdf5"
keras_model = get_unet(do=0.1, activation=ReLU)
keras_model.load_weights(file_path)


# Create the PyTorch model
model, crit, opt = my_unet()  # assuming the PyTorch model is defined as above


# Copy the weights from Keras to PyTorch
model.conv1[0].weight.data = torch.from_numpy(
   np.transpose(keras_model.get_layer("conv2d").weights[0])
)
model.conv1[0].bias.data = torch.from_numpy(
   np.transpose(keras_model.get_layer("conv2d").weights[1])
)
model.conv1[3].weight.data = torch.from_numpy(
   np.transpose(keras_model.get_layer("conv2d_1").weights[0])
)
model.conv1[3].bias.data = torch.from_numpy(
   np.transpose(keras_model.get_layer("conv2d_1").weights[1])
)

model.conv2[0].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_2").weights[0]))
model.conv2[0].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_2").weights[1]))
model.conv2[3].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_3").weights[0]))
model.conv2[3].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_3").weights[1]))

model.conv3[0].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_4").weights[0]))
model.conv3[0].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_4").weights[1]))
model.conv3[3].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_5").weights[0]))
model.conv3[3].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_5").weights[1]))

model.conv4[0].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_6").weights[0]))
model.conv4[0].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_6").weights[1]))
model.conv4[3].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_7").weights[0]))
model.conv4[3].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_7").weights[1]))

model.conv5[0].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_8").weights[0]))
model.conv5[0].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_8").weights[1]))
model.conv5[3].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_9").weights[0]))
model.conv5[3].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_9").weights[1]))

model.up6.weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose").weights[0]))
model.up6.bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose").weights[1]))

model.conv6[0].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_10").weights[0]))
model.conv6[0].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_10").weights[1]))
model.conv6[3].weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_11").weights[0]))
model.conv6[3].bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_11").weights[1]))

model.up7.weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_1").weights[0]))
model.up7.bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_1").weights[1]))

model.conv7[0].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_12").weights[0]))
model.conv7[0].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_12").weights[1]))
model.conv7[3].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_13").weights[0]))
model.conv7[3].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_13").weights[1]))

model.up8.weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_2").weights[0]))
model.up8.bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_2").weights[1]))

model.conv8[0].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_14").weights[0]))
model.conv8[0].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_14").weights[1]))
model.conv8[3].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_15").weights[0]))
model.conv8[3].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_15").weights[1]))

model.up9.weight.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_3").weights[0]))
model.up9.bias.data = torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_transpose_3").weights[1]))

model.conv9[0].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_16").weights[0]))
model.conv9[0].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_16").weights[1]))
model.conv9[3].weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_17").weights[0]))
model.conv9[3].bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_17").weights[1]))

model.conv10.weight.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_18").weights[0]))
model.conv10.bias.data =torch.from_numpy(np.transpose(keras_model.get_layer("conv2d_18").weights[1]))

# Freeze the encoder weights
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.conv2.parameters():
    param.requires_grad = False
for param in model.conv3.parameters():
    param.requires_grad = False
for param in model.conv4.parameters():
    param.requires_grad = False
for param in model.conv5.parameters():
    param.requires_grad = False

torch.save(model.state_dict(), "torchmodel.state_dict.pt")
