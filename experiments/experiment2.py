import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

from data.dataset import get_dataloader
from models.model import SimpleCNN
from models.utils import train

def run_experiment2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(batch_size=32)
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    writer = SummaryWriter(log_dir='./logs/experiment2')
    mlflow.start_run(run_name="experiment2")
    
    for epoch in range(1, 11):
        for batch_idx, batch_size, dataset_size, progress, loss in train(model, device, train_loader, optimizer, epoch):
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
            mlflow.log_metric("loss", loss, step=epoch * len(train_loader) + batch_idx)
    
    writer.close()
    mlflow.pytorch.log_model(model, "model_experiment2")
    mlflow.end_run()
