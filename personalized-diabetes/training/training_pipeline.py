import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import os
from model import GlucoseModel
from gMSE import gMSE

EXPERIMENT_FOLDER_DICT = {
    1: "baseline_1",
    2: "baseline_2",
    3: "baseline_3",
    4: "baseline_4",
    5: "final_model"
}

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
                max_epochs,
                record_losses=False):
    """
    A generic training loop for one phase (self-supervised or supervised).
    Early stopping based on validation loss.
    If record_losses=True, return (train_loss_history, val_loss_history).
    Otherwise, just return the best_val_loss.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = deepcopy(model.state_dict())

    train_loss_history = []
    val_loss_history = []

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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        model.load_state_dict(best_model_state)
                        scheduler.step(val_loss)
                        # If we are recording losses, append the final epoch info
                        if record_losses:
                            epoch_train_loss = running_loss/steps if steps>0 else float('inf')
                            train_loss_history.append(epoch_train_loss)
                            val_loss_history.append(val_loss)
                        return (train_loss_history, val_loss_history) if record_losses else best_val_loss
                scheduler.step(val_loss)

        # End of epoch
        epoch_train_loss = running_loss/steps if steps>0 else float('inf')
        val_loss = evaluate(model, val_loader, criterion, device)

        if record_losses:
            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                model.load_state_dict(best_model_state)
                scheduler.step(val_loss)
                return (train_loss_history, val_loss_history) if record_losses else best_val_loss
        scheduler.step(val_loss)

    model.load_state_dict(best_model_state)
    return (train_loss_history, val_loss_history) if record_losses else best_val_loss


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
    lr_self_sup = trial.suggest_float("lr_self_sup", 1e-5, 1e-2, log=True)
    lr_supervised = trial.suggest_float("lr_supervised", 1e-5, 1e-2, log=True)
    lr_finetune = trial.suggest_float("lr_finetune", 1e-6, 1e-3, log=True)

    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    epochs_self_sup = trial.suggest_int("epochs_self_sup", 1, 20)
    epochs_supervised = trial.suggest_int("epochs_supervised", 1, 20)
    epochs_finetune = trial.suggest_int("epochs_finetune", 1, 20)
    early_stop_patience = trial.suggest_int("early_stop_patience", 0, 5)
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
    n_trials: int = 10,
    eval_frequency: int = 5,
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


def bootstrap_loss(y_true, y_pred, loss_fn, n_boot=1000, confidence=0.95):
    """
    Bootstraps the given predictions to compute a confidence interval
    for the loss. Returns (mean_loss, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed=42)  # for reproducibility
    indices = np.arange(len(y_true))
    
    boot_losses = []
    for _ in range(n_boot):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_pred[sample_idx]
        loss_val = loss_fn(torch.tensor(y_true_sample, dtype=torch.float32),
                           torch.tensor(y_pred_sample, dtype=torch.float32)).item()
        boot_losses.append(loss_val)
    
    boot_losses = np.array(boot_losses)
    lower = np.percentile(boot_losses, ((1.0 - confidence) / 2) * 100)
    upper = np.percentile(boot_losses, (1.0 - (1.0 - confidence) / 2) * 100)
    return float(np.mean(boot_losses)), float(lower), float(upper)

def train_full_with_params(
    best_params,
    X_train_val, Y_train_val,
    X_test, Y_test,
    X_train_self, Y_train_self,
    X_val_self,  Y_val_self,
    X_test_self, Y_test_self,
    self_sup,
    individualized_finetuning,
    patient_id=None,
    
):
    """
    Given the best hyperparameters, trains a fresh model from scratch
    on (train + val) data. If self_sup is True, does a self-supervised
    phase first. Then evaluates on the test set. Finally, we bootstrap 
    the predictions to get a 95% confidence interval for the loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract hyperparams
    lr_self_sup     = best_params["lr_self_sup"]
    lr_supervised   = best_params["lr_supervised"]
    lr_finetune     = best_params["lr_finetune"]
    dropout_rate    = best_params["dropout_rate"]
    epochs_self_sup = best_params["epochs_self_sup"]
    epochs_supervised = best_params["epochs_supervised"]
    epochs_finetune   = best_params["epochs_finetune"]
    early_stop_patience = best_params["early_stop_patience"]

    # Hardcode batch size and eval_frequency for demonstration.
    # You can also incorporate them as best_params if you want.
    batch_size      = 32
    eval_frequency  = 50

    # Fixed hyperparameters for the CNN model
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

    # The code in "objective" used an assertion that the input length is 288
    #   after dropping the "DeidentID" column, so let's do the same check here.
    input_length = int((X_train_val.shape[1] - 1) / 4)  # minus 1 for DeidentID
    assert input_length == 288, "Input length should be 288"

    # ===== Create the Model =====
    model = GlucoseModel(CONV_INPUT_LENGTH=input_length, 
                         self_sup=self_sup, 
                         fixed_hyperparameters=fixed_hyperparameters).to(device)
    criterion = gMSE
    phase_history = {
        "self_supervised": None,
        "supervised": None,
        "finetuning": {}
    }

    # ====== 1) (Optional) Self-Supervised Phase ======
    if self_sup and (X_train_self is not None) and (Y_train_self is not None) \
                 and (X_val_self is not None) and (Y_val_self is not None):
        
        train_loader_ss, val_loader_ss = create_dataloaders(
            X_train_self, Y_train_self,
            X_val_self,   Y_val_self,
            batch_size,
            self_sup=True
        )
        
        optimizer_ss = optim.Adam(model.parameters(), lr=lr_self_sup)
        scheduler_ss = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ss, 'min', patience=2, factor=0.5)

        train_loss_ss, val_loss_ss = train_phase(
            model, 
            criterion, 
            optimizer_ss, 
            scheduler_ss, 
            train_loader_ss, 
            val_loader_ss, 
            device, 
            early_stop_patience, 
            eval_frequency, 
            max_epochs=epochs_self_sup,
            record_losses=True
        )
        phase_history["self_supervised"] = (train_loss_ss, val_loss_ss)
        
        # Switch to supervised mode
        model.self_sup = False
        model.fc5 = nn.Linear(64, 1).to(device)

    # ====== 2) Supervised Phase on (train+val) ======
    train_loader, _ = create_dataloaders(
        X_train_val, Y_train_val,  # (val loader not strictly needed for final training,
        X_train_val, Y_train_val,  #  but we can pass the same data if we still want
                                   #  to do early-stopping or scheduling)
        batch_size,
        self_sup=False
    )
    optimizer_sup = optim.Adam(model.parameters(), lr=lr_supervised)
    scheduler_sup = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sup, 'min', patience=3, factor=0.5)

    train_loss_sup, val_loss_sup = train_phase(
        model, 
        criterion, 
        optimizer_sup, 
        scheduler_sup, 
        train_loader, 
        train_loader,  # if you still want to do early stop with train_loader as val
        device, 
        early_stop_patience, 
        eval_frequency, 
        max_epochs=epochs_supervised,
        record_losses=True
    )
    phase_history["supervised"] = (train_loss_sup, val_loss_sup)
    
    # ====== 3) (Optional) Individualized Fine-Tuning ======
    if individualized_finetuning:
        # If multipatient = True + individualized_finetuning, we do
        # per-patient fine-tuning. We'll do it across all patients in the test set.
        # Or if single patient, it's just that one patient.
        patients_in_data = X_train_val["DeidentID"].unique()
        # We'll store a dictionary of final losses if we want
        final_losses = {}
        
        for pid in patients_in_data:
            phase_history["finetuning"][pid] = None
            # Clone the model so we don't overwrite the global version
            model_clone = deepcopy(model)
            # Pull data for this single patient from train+val
            X_train_val_pid = X_train_val[X_train_val["DeidentID"] == pid]
            Y_train_val_pid = Y_train_val[X_train_val["DeidentID"] == pid]

            train_loader_pid, _ = create_dataloaders(
                X_train_val_pid, Y_train_val_pid,
                X_train_val_pid, Y_train_val_pid,
                batch_size,
                self_sup=False
            )

            optimizer_fine = optim.Adam(model_clone.parameters(), lr=lr_finetune)
            scheduler_fine = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fine, 'min', patience=2, factor=0.5)
            
            train_loss_ft, val_loss_ft = train_phase(
                model_clone,
                criterion,
                optimizer_fine,
                scheduler_fine,
                train_loader_pid,
                train_loader_pid,
                device,
                early_stop_patience,
                eval_frequency,
                max_epochs=epochs_finetune,
                record_losses=True
            )
            phase_history["finetuning"][pid] = (train_loss_ft, val_loss_ft)
            
            # Evaluate on the test set for just that patient
            X_test_pid = X_test[X_test["DeidentID"] == pid]
            Y_test_pid = Y_test[X_test["DeidentID"] == pid]

            # Make predictions
            if len(X_test_pid) == 0:
                continue
            model_clone.eval()
            with torch.no_grad():
                X_test_pid_copy = X_test_pid.drop(columns=["DeidentID"]).values
                X_test_pid_tensor = torch.tensor(X_test_pid_copy, dtype=torch.float32).reshape(-1,1, X_test_pid_copy.shape[1]).to(device)
                preds_pid = model_clone(X_test_pid_tensor).cpu().numpy().flatten()
            
            # Compute the test loss
            test_loss_pid = gMSE(
                torch.tensor(Y_test_pid.drop(columns=["DeidentID"]).values, dtype=torch.float32),
                torch.tensor(preds_pid, dtype=torch.float32)
            ).item()
            final_losses[pid] = test_loss_pid

        print("Per-Patient Fine-Tuning Losses:", final_losses)
        # If you want a single summary metric, you could do:
        avg_loss = np.mean(list(final_losses.values()))
        print(f"Average Test Loss across all patients (fine-tuned): {avg_loss:.4f}")

        # Return or print final losses
        # TODO: bootstrapping and so on for final_model
        return
    
        # ========== Save Plots of Each Phase's Losses ==========
    # We'll put them in "results/<experiment_folder>/" 
    experiment_folder = EXPERIMENT_FOLDER_DICT.get(baseline, "unknown_experiment")
    save_dir = os.path.join("results", experiment_folder)
    os.makedirs(save_dir, exist_ok=True)

    # If single-patient, we might suffix filenames with the patient ID.
    pid_str = f"_patient_{patient_id}" if patient_id else "_multipatient"

    # -- Plot Self-Supervised + Supervised in ONE figure (if needed) --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Loss Curves{pid_str}")

    # Left subplot: self-supervised (if it happened)
    if phase_history["self_supervised"] is not None:
        (ss_train, ss_val) = phase_history["self_supervised"]
        axes[0].plot(ss_train, label="SelfSup Train", marker='o')
        axes[0].plot(ss_val, label="SelfSup Val", marker='x')
        axes[0].set_title("Self-Supervised Phase")
        axes[0].legend()
    else:
        axes[0].set_title("No Self-Supervised Phase")

    # Right subplot: supervised
    (sup_train, sup_val) = phase_history["supervised"]
    axes[1].plot(sup_train, label="Supervised Train", marker='o')
    axes[1].plot(sup_val,   label="Supervised Val", marker='x')
    axes[1].set_title("Supervised Phase")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_phases{pid_str}.png"))
    plt.close(fig)

    # -- If fine-tuning was done, save plots per patient --
    if individualized_finetuning:
        for pid, (ft_train, ft_val) in phase_history["finetuning"].items():
            if ft_train is None:
                continue

            plt.figure(figsize=(6,4))
            plt.plot(ft_train, label="Fine-Tune Train", marker='o')
            plt.plot(ft_val,   label="Fine-Tune Val", marker='x')
            plt.title(f"Fine-Tuning Loss (Patient {pid}){pid_str}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            ft_plot_name = f"finetuning_curve_patient_{pid}.png"
            plt.savefig(os.path.join(save_dir, ft_plot_name))
            plt.close()

    # ====== 4) Evaluate on Test Set + Bootstrap Confidence Interval ======
    # Evaluate using the final model (non-fine-tuned or globally fine-tuned).
    if individualized_finetuning:
        return # We already did this above for each patient
    model.eval()
    with torch.no_grad():
        # Drop DeidentID from test for the actual model input
        X_test_copy = X_test.drop(columns=["DeidentID"]).values
        # Shape => (batch, 1, input_length)
        X_test_tensor = torch.tensor(X_test_copy, dtype=torch.float32).reshape(-1,1,X_test_copy.shape[1]).to(device)
        preds = model(X_test_tensor).cpu().numpy().flatten()
    
    # Actual ground truth from Y_test
    # The glucose column might be shaped (N,) or (N, 1). Just flatten as needed.
    Y_test_copy = Y_test.drop(columns=["DeidentID"]).values.flatten()

    # Compute the test loss (gMSE)
    test_loss = gMSE(
        torch.tensor(Y_test_copy, dtype=torch.float32),
        torch.tensor(preds, dtype=torch.float32)
    ).item()

    # ----- BOOTSTRAP -----
    mean_boot, ci_lower, ci_upper = bootstrap_loss(Y_test_copy, preds, gMSE, n_boot=1000, confidence=0.95)

    # Identify the patient in logs if single-patient approach
    pid_str = f"[Patient {patient_id}] " if patient_id else ""
    print(f"{pid_str}Test loss: {test_loss:.4f}")
    print(f"{pid_str}Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (Mean bootstrapped loss: {mean_boot:.4f})")
