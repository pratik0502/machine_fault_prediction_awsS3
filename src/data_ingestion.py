import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv(r'D:\ML_ops\pratik\awsS3+dvc\data\raw\data.csv')

# # Separating features and target variable
# X = df.drop(columns=['fail'])
# y = df['fail']

# # Scaling the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Applying PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Creating a DataFrame with PCA results
# df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
# df_pca['fail'] = y.values

# df_pca.to_csv(os.path.join('data','processed','machine_fault_pca.csv'), index=False)

x = df.drop(columns=['fail'])
y = df['fail']

sca = StandardScaler()
X_scaled = sca.fit_transform(x)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca,columns=['pc1','pc2'])
df_pca['fail'] = y.values

df_pca.to_csv(os.path.join('data','processed','machine_fault_pca.csv'),index=False)