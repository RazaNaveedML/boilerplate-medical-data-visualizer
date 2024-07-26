import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv('medical_examination.csv')

# 2. Create the overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize cholesterol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Function to draw categorical plot
def draw_cat_plot():
    # 5. Convert data to long format
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Prepare data for the cat plot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7. Draw the cat plot
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='bar', height=4, aspect=1.2)
    
    # Set labels
    for ax in fig.axes.flat:
        ax.set_xlabel('variable')
        ax.set_ylabel('total')
    
    # 8. Get the figure
    fig = fig.fig
    
    # 9. Save the figure
    fig.savefig('catplot.png')
    return fig

# 10. Function to draw heat map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # 12. Calculate the correlation matrix
    corr = df_heat.corr()
    
    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 15. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
    
    # 16. Save the figure
    fig.savefig('heatmap.png')
    return fig
