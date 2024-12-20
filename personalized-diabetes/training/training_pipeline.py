import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
from copy import deepcopy
import time
from model import GlucoseModel
from gMSE import gMSE


class CustomDataset(Dataset):
    def __init__(self, X, Y=None, self_sup=False):
        """
        X, Y are numpy arrays or pandas DataFrames/Series.
        The model expects input shape: (batch, 1, input_length)
        We'll assume X is already in the correct shape for this purpose.
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.X = X.values
        else:
            self.X = X
        if Y is not None:
            if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
                self.Y = Y.values
            else:
                self.Y = Y
        else:
            self.Y = None
        self.self_sup = self_sup

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Reshape to (1, length_of_signals)
        x = x.reshape(1, -1).astype(np.float32)
        if self.Y is not None:
            y = self.Y[idx].astype(np.float32)
            return x, y
        else:
            return x


def create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size, self_sup=False):
    X_train_copy = X_train.drop(columns=["DeidentID"])
    X_val_copy = X_val.drop(columns=["DeidentID"])
    Y_train_copy = Y_train.drop(columns=["DeidentID"])
    Y_val_copy = Y_val.drop(columns=["DeidentID"])
    train_dataset = CustomDataset(X_train_copy, Y_train_copy, self_sup=self_sup)
    val_dataset = CustomDataset(X_val_copy, Y_val_copy, self_sup=self_sup)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_phase(model, 
                criterion, 
                optimizer, 
                scheduler, 
                train_loader, 
                val_loader, 
                device, 
                early_stop_patience, 
                eval_frequency, 
                max_epochs):
    """
    A generic training loop for one phase (self-supervised or supervised).
    Early stopping based on validation loss.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = deepcopy(model.state_dict())  # To handle the case of zero epochs

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        
        for i, batch in enumerate(train_loader):
            if len(batch) == 2:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            else:
                x_batch = batch[0].to(device)
                y_batch = None

            optimizer.zero_grad()
            outputs = model(x_batch)
            if y_batch is not None:
                loss = criterion(y_batch, outputs) 
            else:
                loss = criterion(outputs)  
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            steps += 1
            
            # Evaluate after some steps
            if (i+1) % eval_frequency == 0:
                val_loss = evaluate(model, val_loader, criterion, device)
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        # Early stop
                        model.load_state_dict(best_model_state)
                        scheduler.step(val_loss)
                        return best_val_loss
                # Step the scheduler after validation
                scheduler.step(val_loss)

        # End of epoch validation
        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                model.load_state_dict(best_model_state)
                scheduler.step(val_loss)
                return best_val_loss
        scheduler.step(val_loss)

    model.load_state_dict(best_model_state)
    return best_val_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            else:
                x_batch = batch[0].to(device)
                y_batch = None

            outputs = model(x_batch)
            if y_batch is not None:
                loss = criterion(y_batch, outputs)
            else:
                loss = criterion(outputs)
            total_loss += loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else float('inf')


def objective(trial,
              X_train, Y_train,
              X_val, Y_val,
              X_test, Y_test,
              X_train_self, Y_train_self,
              X_val_self, Y_val_self,
              X_test_self, Y_test_self,
              self_sup,
              individualized_finetuning,
              eval_frequency):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters from Optuna
    lr_self_sup = trial.suggest_float("lr_self_sup", 1e-4, 1e-2, log=True)
    lr_supervised = trial.suggest_float("lr_supervised", 1e-4, 1e-2, log=True)
    lr_finetune = trial.suggest_float("lr_finetune", 1e-5, 1e-3, log=True)

    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    epochs_self_sup = trial.suggest_int("epochs_self_sup", 1, 10)
    epochs_supervised = trial.suggest_int("epochs_supervised", 1, 10)
    epochs_finetune = trial.suggest_int("epochs_finetune", 1, 10)
    early_stop_patience = trial.suggest_int("early_stop_patience", 1, 5)
    batch_size = 32

    # Fixed hyperparameters for convolutional layers
    fixed_hyperparameters = {
        "filter_1": 4,
        "kernel_1": 6,
        "stride_1": 2,
        "pool_size_1": 2,
        "pool_stride_1": 2,
        "filter_2": 7,
        "kernel_2": 5,
        "stride_2": 2,
        "pool_size_2": 6,
        "pool_stride_2": 5,
        "dropout_rate": dropout_rate
    }

    # Assuming input length matches what was done previously
    input_length = int((X_train.shape[1] - 1) / 4) # should be (batch, input_length) after reshape, -1 is due to patient id
    assert input_length == 288, "Input length should be 288"
    model = GlucoseModel(CONV_INPUT_LENGTH=input_length, self_sup=self_sup, fixed_hyperparameters=fixed_hyperparameters).to(device)
    criterion = gMSE
    patients = X_train["DeidentID"].unique().tolist()
    # ============= Self-Supervised Phase (Optional) =============
    if self_sup and X_train_self is not None and Y_train_self is not None and X_val_self is not None and Y_val_self is not None:
        # Self-supervised training
        train_loader_ss, val_loader_ss = create_dataloaders(X_train_self, Y_train_self, X_val_self, Y_val_self, batch_size, self_sup=True)
        optimizer_ss = optim.Adam(model.parameters(), lr=lr_self_sup)
        scheduler_ss = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ss, 'min', patience=2, factor=0.5)

        best_val_loss_ss = train_phase(model, criterion, optimizer_ss, scheduler_ss, train_loader_ss, val_loader_ss,
                                       device, early_stop_patience, eval_frequency, epochs_self_sup)
        # After self-supervised training, switch to supervised mode
        model.self_sup = False
        model.fc5 = nn.Linear(64, 1).to(device)
    else:
        # If not self_sup, we are already in supervised mode
        pass

    # ============= Supervised Phase =============
    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size, self_sup=False)
    optimizer_sup = optim.Adam(model.parameters(), lr=lr_supervised)
    scheduler_sup = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sup, 'min', patience=3, factor=0.5)

    best_val_loss_sup = train_phase(model, criterion, optimizer_sup, scheduler_sup, train_loader, val_loader,
                                    device, early_stop_patience, eval_frequency, epochs_supervised)

    # ============= Individualized Fine-Tuning (Optional) =============
    if individualized_finetuning:
        final_metrics = []
        for patient in patients:
            model_clone = deepcopy(model)
            X_train_patient = X_train[X_train["DeidentID"] == patient]
            Y_train_patient = Y_train[X_train["DeidentID"] == patient]
            X_val_patient = X_val[X_val["DeidentID"] == patient]
            Y_val_patient = Y_val[X_val["DeidentID"] == patient]
            
            train_loader_patient, val_loader_patient = create_dataloaders(X_train_patient, Y_train_patient, X_val_patient, Y_val_patient, batch_size, self_sup=False)
            optimizer_finetune = optim.Adam(model_clone.parameters(), lr=lr_finetune)
            scheduler_finetune = optim.lr_scheduler.ReduceLROnPlateau(optimizer_finetune, 'min', patience=2, factor=0.5)
            
            best_val_loss_finetune = train_phase(model_clone, criterion, optimizer_finetune, scheduler_finetune, train_loader_patient, val_loader_patient,
                                                device, early_stop_patience, eval_frequency, epochs_finetune)
            
            final_metrics.append(best_val_loss_finetune)
        final_metric = np.mean(final_metrics)
    else:
        final_metric = best_val_loss_sup
    return final_metric


def run_optuna_study(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    X_train_self, Y_train_self,
    X_val_self, Y_val_self,
    X_test_self, Y_test_self,
    self_sup: bool,
    individualized_finetuning: bool,
    n_trials: int = 500,
    eval_frequency: int = 50,
    direction: str = "minimize",
):
    """
    Runs an Optuna study using the provided train/val/test sets.
    """
    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: objective(
            trial,
            X_train, Y_train,
            X_val, Y_val,
            X_test, Y_test,
            X_train_self, Y_train_self,
            X_val_self, Y_val_self,
            X_test_self, Y_test_self,
            self_sup,
            individualized_finetuning,
            eval_frequency
        ),
        n_trials=n_trials,
    )
    return study
