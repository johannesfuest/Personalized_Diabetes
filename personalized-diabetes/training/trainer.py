import pandas as pd

DATASET = "basic_0.csv"
DATASET_SELF = "self_0.csv"
missing_modulos = [10, 20, 50, 100, 200, 400, 800, 1000, 1500, 2000]

patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]

# TODO: write function that flexibly executes the training loop for any baseline, logs stuff and saves all relevant plots. 

def run_experiment(baseline: int, test: bool):
    multipatient = True
    self_sup = False
    finetune = False
    if baseline == 2:
        self_sup = True
    elif baseline == 3:
        multipatient = True
    elif baseline == 4:
        multipatient = False
        self_sup = True
    else:
        self_sup = True
        finetune = True
        
    df_basic = pd.read_csv(DATASET)
    df_self = pd.read_csv(DATASET_SELF)
    
    df_basic = df_basic[~df_basic.DeidentID.isin(patients_to_exclude)]
    df_self = df_self[~df_self.DeidentID.isin(patients_to_exclude)]
    
    
        