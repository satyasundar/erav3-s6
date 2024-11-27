from model import Network, train, test, device
import torch.optim as optim
from torchvision import datasets, transforms
import torch

def main():
    # Model
    model = Network().to(device)
    
    # Data loading
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomRotation((-15., 15.)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=128, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=128, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(20):
        print(f'Epoch {epoch+1}')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main() 