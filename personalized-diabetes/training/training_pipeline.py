import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
from copy import deepcopy
import time
from utils import get_train_test_split_across_patients, get_train_test_split_single_patient
from model import GlucoseModel
from gMSE import gMSE

# Assuming gMSE is defined somewhere else
# from gMSE import gMSE

class CustomDataset(Dataset):
    def __init__(self, X, Y=None, self_sup=False):
        """
        X, Y are pandas DataFrames.
        The model expects input shape: (batch, 1, CONV_INPUT_LENGTH*4)
        We'll assume X is already in the correct order of features 
        or that we know how to reshape them.
        
        If self_sup=True, Y might not be typical CGM; 
        it could be self-supervised targets (4-dimensional, etc.).
        """
        self.X = X.values
        self.Y = None if Y is None else Y.values
        self.self_sup = self_sup

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Reshape to (1, length_of_signals)
        # For example if X has 4 segments concatenated, 
        # and each segment is CONV_INPUT_LENGTH long,
        # length_of_signals = 4*CONV_INPUT_LENGTH
        x = x.reshape(1, -1).astype(np.float32)
        if self.Y is not None:
            y = self.Y[idx].astype(np.float32)
            return x, y
        else:
            return x


def create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size, self_sup=False):
    train_dataset = CustomDataset(X_train, Y_train, self_sup=self_sup)
    val_dataset = CustomDataset(X_val, Y_val, self_sup=self_sup)

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
            loss = criterion(y_batch, outputs) if y_batch is not None else criterion(outputs)  # depending on how gMSE is defined
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
                        scheduler.step(val_loss)  # allow scheduler step on stop
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
            loss = criterion(y_batch, outputs) if y_batch is not None else criterion(outputs)
            total_loss += loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else float('inf')


def objective(trial, 
              df,
              df_ss=None,
              multi_patient=True,
              self_sup=False,
              individualized_finetuning=False,
              patients=[1,2,3],
              TRAIN_TEST_SPLIT=0.8,
              eval_frequency=50):
    """
    An Optuna objective function that:
    - Optionally performs self-supervised pre-training
    - Then supervised training
    - Optionally does individualized fine-tuning at the end if multi-patient
    
    Hyperparameters to tune:
    - learning rates for each phase
    - dropout rates
    - number of epochs for each phase
    - early stopping patience
    - batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters from Optuna
    # Note: You can adjust the search ranges as desired.
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
        "filter_1": 3,
        "kernel_1": 6,
        "stride_1": 2,
        "pool_size_1": 3,
        "pool_stride_1": 2,
        "filter_2": 7,
        "kernel_2": 6,
        "stride_2": 2,
        "pool_size_2": 6,
        "pool_stride_2": 4,
        "dropout_rate": dropout_rate
    }

    # Filter the dataframe by the selected patients
    df_filtered = df[df['DeidentID'].isin(patients)]
    if self_sup and df_ss is not None:
        df_ss_filtered = df_ss[df_ss['DeidentID'].isin(patients)]
    else:
        df_ss_filtered = None

    # Decide how to split
    if multi_patient:
        X_train, X_val, Y_train, Y_val = get_train_test_split_across_patients(df_filtered, TRAIN_TEST_SPLIT, self_sup=False)
        if self_sup and df_ss_filtered is not None:
            X_train_ss, X_val_ss, Y_train_ss, Y_val_ss = get_train_test_split_across_patients(df_ss_filtered, TRAIN_TEST_SPLIT, self_sup=True)
        else:
            X_train_ss = X_val_ss = Y_train_ss = Y_val_ss = None
    else:
        # If single patient, we expect patients to have length 1
        patient_id = patients[0]
        df_single = df_filtered[df_filtered['DeidentID'] == patient_id]
        X_train, X_val, Y_train, Y_val = get_train_test_split_single_patient(df_single, TRAIN_TEST_SPLIT, self_sup=False)
        if self_sup and df_ss_filtered is not None:
            df_ss_single = df_ss_filtered[df_ss_filtered['DeidentID'] == patient_id]
            X_train_ss, X_val_ss, Y_train_ss, Y_val_ss = get_train_test_split_single_patient(df_ss_single, TRAIN_TEST_SPLIT, self_sup=True)
        else:
            X_train_ss = X_val_ss = Y_train_ss = Y_val_ss = None

   
    input_length = 288 * 4

    # Initialize the model
    model = GlucoseModel(CONV_INPUT_LENGTH=input_length, self_sup=self_sup, fixed_hyperparameters=fixed_hyperparameters).to(device)
    criterion = gMSE

    # ============= Self-Supervised Phase (Optional) =============
    if self_sup and X_train_ss is not None:
        # Create dataloaders for self-supervised
        train_loader_ss, val_loader_ss = create_dataloaders(X_train_ss, Y_train_ss, X_val_ss, Y_val_ss, batch_size, self_sup=True)

        optimizer_ss = optim.Adam(model.parameters(), lr=lr_self_sup)
        scheduler_ss = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ss, 'min', patience=2, factor=0.5)

        # Train self-supervised
        best_val_loss_ss = train_phase(model, criterion, optimizer_ss, scheduler_ss, train_loader_ss, val_loader_ss,
                                       device, early_stop_patience, eval_frequency, epochs_self_sup)

        # After self-supervised, change last layer to output a single value
        # and switch self_sup off in model
        model.self_sup = False
        model.fc5 = nn.Linear(64, 1).to(device)

    # ============= Supervised Phase =============
    # Create supervised dataloaders
    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size, self_sup=False)

    optimizer_sup = optim.Adam(model.parameters(), lr=lr_supervised)
    scheduler_sup = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sup, 'min', patience=3, factor=0.5)

    best_val_loss_sup = train_phase(model, criterion, optimizer_sup, scheduler_sup, train_loader, val_loader,
                                    device, early_stop_patience, eval_frequency, epochs_supervised)

    # ============= Individualized Fine-Tuning (Optional) =============
    # If multi-patient and individualized_finetuning, we fine-tune a new model for each patient individually
    # and measure performance.
    # For simplicity, we’ll just do a quick pass and return average fine-tune performance.
    if multi_patient and individualized_finetuning:
        fine_tune_losses = []
        base_model_state = deepcopy(model.state_dict())

        for pid in patients:
            # Filter data for just this patient
            df_p = df[df['DeidentID'] == pid]
            X_train_p, X_val_p, Y_train_p, Y_val_p = get_train_test_split_single_patient(df_p, TRAIN_TEST_SPLIT, self_sup=False)
            train_loader_p, val_loader_p = create_dataloaders(X_train_p, Y_train_p, X_val_p, Y_val_p, batch_size, self_sup=False)

            # Reset model to base supervised trained weights
            model.load_state_dict(base_model_state)

            # Lower learning rate for fine-tuning
            optimizer_ft = optim.Adam(model.parameters(), lr=lr_finetune)
            scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', patience=2, factor=0.5)

            # Short fine-tune phase
            val_loss_ft = train_phase(model, criterion, optimizer_ft, scheduler_ft, train_loader_p, val_loader_p,
                                      device, early_stop_patience, eval_frequency, epochs_finetune)
            fine_tune_losses.append(val_loss_ft)

        # Return average fine-tune loss as metric
        final_metric = np.mean(fine_tune_losses)
    else:
        # If no fine-tuning, return the supervised val loss
        final_metric = best_val_loss_sup

    return final_metric
