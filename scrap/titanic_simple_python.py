from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

titanic = pd.read_csv('data/titanic.csv')
print(f'Loaded data/titanic.csv as titanic: {len(titanic)} rows, {len(titanic.columns)} columns')
print('\nDataset Info for titanic:')
titanic.info()
print('\nDescriptive Statistics for titanic:')
print(titanic.describe())
clean_data = titanic.copy()
clean_data['Age'] = clean_data['Age'].fillna(30)

print('Filled NA values in titanic')
survivors = clean_data[clean_data['Survived'] == 1].copy()
print(f'Filtered clean_data: {len(survivors)} rows match condition')
print('\nDataset Summary for survivors:')
print(f'Shape: {survivors.shape}')
print(f'Columns: {list(survivors.columns)}')
print('\nData types:')
print(survivors.dtypes)
print('\nMissing values:')
print(survivors.isnull().sum())
by_gender = clean_data.groupby(['Sex']).size().reset_index(name='count')
print(f"Grouped by ['Sex']: {len(by_gender)} groups")
# Box plot
plt.figure(figsize=(10, 6))
clean_data.boxplot(column='Age', by='Pclass')
plt.title('Box Plot')
plt.xticks(rotation=45)


# Display plots
plt.tight_layout()
try:
    get_ipython()
    # Running in Jupyter/IPython - don't show (kernel will display inline)
except NameError:
    # Running in VS Code/CLI - show plots in separate windows
    plt.show()