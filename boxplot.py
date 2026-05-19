import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('compare_model.csv')
metrics = ['ARI', 'NMI', 'HOM']

for metric in metrics:
    plt.figure(figsize=(8, 5))
    
    sns.boxplot(
        data=df,
        x='Method',
        y=metric
    )
    '''
    sns.stripplot(
        data=df,
        x='Method',
        y=metric,
        alpha=0.6
    )
    '''

    plt.title(f'{metric} Distribution by Method')
    plt.ylabel(metric)
    plt.xlabel('Method')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.show()

# Create 3 plots in ONE figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(metrics):

    sns.boxplot(
        data=df,
        x='Method',
        y=metric,
        ax=axes[i]
    )

    '''
    sns.stripplot(
        data=df,
        x='Method',
        y=metric,
        ax=axes[i],
        alpha=0.6
    )
    '''

    axes[i].set_title(f'{metric} Distribution')
    axes[i].set_xlabel('Method')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('compare_model_boxplot.png', dpi=300)
plt.show()