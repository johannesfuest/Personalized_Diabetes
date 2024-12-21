import pandas as pd
from training_pipeline import run_optuna_study, train_full_with_params
from utils import get_train_test_split_across_patients, get_train_test_split_single_patient

DATASET = "basic_0.csv"
DATASET_SELF = "self_0.csv"
missing_modulos = [10, 20, 50, 100, 200, 400, 800, 1000, 1500, 2000]

patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
patients = range(1, 31)
patients = [p for p in patients if p not in patients_to_exclude]


def run_experiment(baseline: int, test: bool):
    
    if baseline == 1:
        multipatient = True
        self_sup = False
        finetune = False
    elif baseline == 2:
        multipatient = True
        self_sup = True
        finetune = False
    elif baseline == 3:
        multipatient = False
        self_sup = False
        finetune = False
    elif baseline == 4:
        multipatient = False
        self_sup = True
        finetune = False
    else:
        multipatient = True
        self_sup = True
        finetune = True
        
    df_basic = pd.read_csv(DATASET)
    df_self = pd.read_csv(DATASET_SELF)
    
    df_basic = df_basic[df_basic.DeidentID.isin(patients)]
    df_self = df_self[df_self.DeidentID.isin(patients)]
    
    
    def keep_first_n_rows_per_id(df, id_col='DeidentID', n=2000):
        return df.groupby(id_col, group_keys=False).head(n)

    
    if test:
        df_basic = keep_first_n_rows_per_id(df_basic)
        keys_in_basic = set(df_basic[['LocalDtTm', 'DeidentID']].apply(tuple, axis=1))
        df_self = df_self[df_self[['LocalDtTm', 'DeidentID']].apply(tuple, axis=1).isin(keys_in_basic)]
        
    if multipatient:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = get_train_test_split_across_patients(df_basic, 0.8, False)
        
        if self_sup:
            X_train_self, X_val_self, X_test_self, Y_train_self, Y_val_self, Y_test_self = get_train_test_split_across_patients(df_self, 0.8, True)
        else:
            X_train_self = None
            Y_train_self = None
            X_val_self = None
            Y_val_self = None
            X_test_self = None
            Y_test_self = None
        study = run_optuna_study(X_train, Y_train, X_val, Y_val, X_test, Y_test, X_train_self, Y_train_self, X_val_self, Y_val_self, X_test_self, Y_test_self, self_sup, finetune)
        best_params = study.best_params
        print("Best hyperparameters found by Optuna:", best_params)

        # ---- 2) Retrain the model on (train + val) with best hyperparams ----
        # Combine train and val sets
        X_train_val = pd.concat([X_train, X_val])
        Y_train_val = pd.concat([Y_train, Y_val])

        # We'll write a helper function (in training_pipeline.py) called
        # `train_full_with_params` that trains a fresh model from scratch
        # using the best hyperparams, *optionally* does self-supervised pre-training,
        # then returns final model along with the test loss + bootstrap intervals.
        # 
        # We also compute final metrics for each patient in the test set.

        print("\nRetraining model with best hyperparams on [train + val] and evaluating on test...\n")
        train_full_with_params(
            best_params, 
            X_train_val, Y_train_val,
            X_test, Y_test,
            X_train_self, Y_train_self,
            X_val_self,  Y_val_self,
            X_test_self, Y_test_self,
            self_sup, 
            finetune
        )
        
    else:
        for patient in patients:
            df_patient = df_basic[df_basic["DeidentID"] == patient].copy()
            if df_patient.shape[0] == 0:
                print(f"Skipping patient {patient} as they have no data")
                continue
            df_self_patient = df_self[df_self["DeidentID"] == patient].copy()
            X_train, X_val, X_test, Y_train, Y_val, Y_test = get_train_test_split_single_patient(df_patient, 0.8, False)
            if self_sup:
                X_train_self, X_val_self, X_test_self, Y_train_self, Y_val_self, Y_test_self = get_train_test_split_single_patient(df_self_patient, 0.8, self_sup)
            else:
                X_train_self = None
                Y_train_self = None
                X_val_self = None
                Y_val_self = None
                X_test_self = None
                Y_test_self = None
            study = run_optuna_study(X_train, Y_train, X_val, Y_val, X_test, Y_test, X_train_self, Y_train_self, X_val_self, Y_val_self, X_test_self, Y_test_self, self_sup, finetune)
            best_params = study.best_params
            print(f"Patient {patient} best params: {best_params}")

            # Retrain using (train + val)
            X_train_val = pd.concat([X_train, X_val])
            Y_train_val = pd.concat([Y_train, Y_val])

            print(f"\nRetraining model for Patient {patient} with best hyperparams...\n")
            train_full_with_params(
                best_params,
                X_train_val, Y_train_val,
                X_test, Y_test,
                X_train_self, Y_train_self,
                X_val_self,  Y_val_self,
                X_test_self, Y_test_self,
                self_sup,
                finetune,
                patient_id=patient
            )
    
if __name__ == "__main__":
    run_experiment(4, True)