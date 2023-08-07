import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.widgets import Button, CheckButtons
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bitcoin.csv')

plt.figure(figsize=(15, 7))
plt.plot(df['Close'])
plt.title('Bitcoin price.', fontsize=15)
plt.ylabel('Price in dollars.')

plt.show()