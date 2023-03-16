import pandas as pd
col_names = ['id', 'activity time', 'transpo_model',
'sport', 'picnic', 'read', 'walk', 'meditate', 'jog',
'rating', 'playground']
family_fr = pd.read_excel('family activity.xlsx', names = col_names, usecols='B:K',header=0 , sheet_name= 'Formatted data')
#print(family_fr)
chisqt = pd.crosstab(family_fr['rating'], family_fr['playground'], margins=True)
print(chisqt)
import numpy as np
data = np.array(chisqt.iloc[0:4,0:2].values)
#print(data)
from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(data)
print(stat)
print(p)
print(dof)
print(expected)
alpha = 0.05
print('p value is ' + str(p))
if p <= alpha:
    print('dependent (reject H0)')
else:
    print('independent (H0 holds true')