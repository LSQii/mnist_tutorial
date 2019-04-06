from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__()#复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 20张特征图, 卷积核为5x1正方形
        self.conv2 = nn.Conv2d(20, 50, 5, 1)# 定义conv2函数的是图像卷积函数：输入为20张特征图,输出为50张特征图, 卷积核为5x1矩形
        self.fc1 = nn.Linear(4*4*50, 500) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将4*4*50个节点连接到500个节点上。
        self.fc2 = nn.Linear(500, 10)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将500个节点连接到10个节点上。

    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # 输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, 4*4*50) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x))  #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = self.fc2(x) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        return F.log_softmax(x, dim=1) # softmax 分类
#训练函数如下
"""
其中，model指建立好的网络模型；device指模型在哪个设备上运行，CPU还是GPU；train_loader是指数据集；
optimizer用于优化；epoch指整个数据集训练的次数
"""
def train(args, model, device, train_loader, optimizer, epoch,train_correct,train_total):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):# batch_idx为索引，（data, target）为值
        data, target = data.to(device), target.to(device) # data为图像，target为label

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # 利用输出和label计算损失精度
        loss.backward() # 反向传播，用来计算梯度
        optimizer.step() #更新参数

        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum()
    print('train accuracy: %.2f%%' % (100 * train_correct / train_total))

        #if batch_idx % args.log_interval == 0:
         #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          #      epoch, batch_idx * len(data), len(train_loader.dataset),
           #     100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, test_accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_correct = 0
    train_total = 0

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,train_correct, train_total)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()

