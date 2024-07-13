import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

from data.dataset.py import get_dataloader
from models.model import SimpleCNN
from models.utils import train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader()
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters())
    
    writer = SummaryWriter(log_dir='./logs')
    mlflow.start_run()
    
    for epoch in range(1, 11):
        for batch_idx, batch_size, dataset_size, progress, loss in train(model, device, train_loader, optimizer, epoch):
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
            mlflow.log_metric("loss", loss, step=epoch * len(train_loader) + batch_idx)
    
    writer.close()
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == '__main__':
    main()
