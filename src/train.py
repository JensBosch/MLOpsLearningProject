from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
from torch.optim import Adam
import hydra
from datasets.titanic_dataset import TitanicDataset
from models.titanic_prediciton import TitanicPrediction
from omegaconf import DictConfig


def training_step(model: nn.Module, train_loader: DataLoader, optimizer, loss_function):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch
        y_batch = y_batch

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    return train_loss


def eval_step(model: nn.Module, val_loader: DataLoader, loss_function):
    model.eval()
    val_loss = 0.0

    with no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch
            y_batch = y_batch

            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)

    return val_loss

@hydra.main(version_base=None, config_path="conf", config_name="titanic_nn")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    manual_seed(cfg.seed)
    with mlflow.start_run():
        num_epochs = 20
        model = TitanicPrediction(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
        mlflow.pytorch.log_model(
            model,
            name="model"
        )
        dataset = TitanicDataset(cfg.dataset_path)
        optimizer = Adam(model.parameters(), lr=1e-3)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        loss_function = nn.HingeEmbeddingLoss()

        for epoch in tqdm(range(num_epochs)):
            train_loss = training_step(model, train_loader, optimizer, loss_function)
            train_loss = train_loss / len(train_dataset)

            val_loss = eval_step(model, val_loader, loss_function)
            val_loss = val_loss / len(val_dataset)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=cfg.model_name
        )


if __name__ == "__main__":
    train()
