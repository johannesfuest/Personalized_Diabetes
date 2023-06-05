import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# set seaborn styling
sns.set_theme(style="white")

if __name__ == '__main__':
    df_data_summary = pd.read_csv('data_summary.csv')

    #df_data_summary = df_data_summary[df_data_summary['nrows'] > 17000]


    fig, ax = plt.subplots(2, 3, figsize=(20, 15))

    # Create histogram of number of rows per patient
    sns.histplot(df_data_summary['nrows'], bins=14, ax=ax[0, 0], color='darkblue', edgecolor='black')
    ax[0, 0].set_title('Number of rows per patient', fontsize=30)
    ax[0, 0].set_xlabel('Number of rows', fontsize=24)
    ax[0, 0].set_ylabel('Number of patients', fontsize=24)
    ax[0, 0].spines['top'].set_visible(True)
    ax[0, 0].spines['right'].set_visible(True)

    # Create histogram of average CGM per patient
    sns.histplot(df_data_summary['avg_cgm'], bins=14, ax=ax[0, 1], color='darkblue', edgecolor='black')
    ax[0, 1].set_title('Average CGM per patient', fontsize=30)
    ax[0, 1].set_xlabel('Average CGM', fontsize=24)
    ax[0, 1].set_ylabel('Number of patients', fontsize=24)
    ax[0, 1].spines['top'].set_visible(True)
    ax[0, 1].spines['right'].set_visible(True)

    # Create histogram of average insulin per patient
    sns.histplot(df_data_summary['avg_insulin'], bins=14, ax=ax[0, 2], color='darkblue', edgecolor='black')
    ax[0, 2].set_title('Average insulin per patient', fontsize=30)
    ax[0, 2].set_xlabel('Average insulin', fontsize=24)
    ax[0, 2].set_ylabel('Number of patients', fontsize=24)
    ax[0, 2].spines['top'].set_visible(True)
    ax[0, 2].spines['right'].set_visible(True)

    # Create histogram of average exercise per patient
    sns.histplot(df_data_summary['avg_exercise'], bins=14, ax=ax[1, 0], color='darkblue', edgecolor='black')
    ax[1, 0].set_title('Average exercise per patient', fontsize=30)
    ax[1, 0].set_xlabel('Average exercise', fontsize=24)
    ax[1, 0].set_ylabel('Number of patients', fontsize=24)
    ax[1, 0].spines['top'].set_visible(True)
    ax[1, 0].spines['right'].set_visible(True)

    # Create histogram of average meal size per patient
    sns.histplot(df_data_summary['avg_mealsize'], bins=14, ax=ax[1, 1], color='darkblue', edgecolor='black')
    ax[1, 1].set_title('Average meal size per patient', fontsize=30)
    ax[1, 1].set_xlabel('Average meal size', fontsize=24)
    ax[1, 1].set_ylabel('Number of patients', fontsize=24)
    ax[1, 1].spines['top'].set_visible(True)
    ax[1, 1].spines['right'].set_visible(True)

    # Create histogram of average glucose per patient
    sns.histplot(df_data_summary['avg_glucose'], bins=14, ax=ax[1, 2], color='darkblue', edgecolor='black')
    ax[1, 2].set_title('Average glucose per patient', fontsize=30)
    ax[1, 2].set_xlabel('Average glucose', fontsize=24)
    ax[1, 2].set_ylabel('Number of patients', fontsize=24)
    ax[1, 2].spines['top'].set_visible(True)
    ax[1, 2].spines['right'].set_visible(True)

    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    plt.show()

