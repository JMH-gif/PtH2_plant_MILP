import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plot = False
cluster_n = 7*24
# Import Data from the Saxony-Anhalt report
path_SA_report = r'C:\Users\Mika\Desktop\Master\20240119_Strukturen.xlsx'
'''
cf_wind = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="H").values
cf_wind = [cf_wind[i][0] for i in range(len(cf_wind))]
cf_pv = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="C").values
cf_pv = [cf_pv[i][0] for i in range(len(cf_pv))]
'''
df = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols='C, H')
print(df)
sns.set_style('darkgrid')
sns.color_palette(palette='RdYlGn')
if plot == True:
    sns.scatterplot([cf_pv, cf_wind]).set(title='Installed Capacities per technology')
    plt.show()

def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range (1, max_k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    # Generate the elbow plot
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
optimise_k_means(df, cluster_n)

kmeans = KMeans(n_clusters=cluster_n)
kmeans.fit(df)
df['kmeans_48'] = kmeans.labels_
print(df)
plt.scatter(x=df['pv_2'], y = df['wind_2'], c = df['kmeans_48'])
plt.show()
