import gspread
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

gc = gspread.service_account(filename='mycredentials.json')
gsheet = gc.open_by_key("1ieph14IFEyGr2LsDJ8JmGPxreWR8MBgcCn8-7REW_HQ")
mydata = gsheet.sheet1.get_all_records()
df = pd.DataFrame(mydata)
del df['Timestamp']
#  Changing string elements to numbers so the classifier can work
label_encoder = preprocessing.LabelEncoder()
df['¿Cómo resuelves los problemas con tus amigos/familiares? [Respuesta:]'] = label_encoder.fit_transform(
    df['¿Cómo resuelves los problemas con tus amigos/familiares? [Respuesta:]'])
df['¿Con qué tipo de personalidad te identificas? [Respuesta:]'] = label_encoder.fit_transform(
    df['¿Con qué tipo de personalidad te identificas? [Respuesta:]'])
df['Si no estoy de acuerdo contigo en algunos temas, ¿cómo te sientes? [Respuesta:]'] = label_encoder.fit_transform(
    df['Si no estoy de acuerdo contigo en algunos temas, ¿cómo te sientes? [Respuesta:]'])
df['¿Prefieres ver películas o leer libros? [Respuesta:]'] = label_encoder.fit_transform(
    df['¿Prefieres ver películas o leer libros? [Respuesta:]'])
df['Email Address'] = label_encoder.fit_transform(df['Email Address'])
del df['Email Address']
df = df.rename(columns={'¿Qué tan religios@ eres?': 'Religion',
                        '¿Crees que las relaciones a distancia pueden funcionar?': 'Distance',
                        '¿Hablas de tus problemas personales con tus amigos cercanos?': 'Personal',
                        '¿Eres propenso a tomar decisiones impulsivas?': 'Implusive',
                        '¿Quieres casarte?': 'Marry',
                        '¿Crees en los horóscopos y signos del zodiaco?': 'Zodiac',
                        '¿Qué tan consciente eres de tu salud?': 'Health',
                        '¿Te gusta hablar sobre tus problemas?': 'Problems',
                        '¿Tu ego te impide disculparte cuando te equivocas?': 'Ego',
                        '¿Cómo resuelves los problemas con tus amigos/familiares? [Respuesta:]': 'Family',
                        '¿Con qué tipo de personalidad te identificas? [Respuesta:]': 'Personality',
                        '¿Haces ejercicio?': 'Gym',
                        'Si no estoy de acuerdo contigo en algunos temas, ¿cómo te sientes? [Respuesta:]': 'Agree',
                        '¿Prefieres ver películas o leer libros? [Respuesta:]': 'Movies',
                        'Sexo': 'Sex'})
# print(df.head())
features = ['Religion', 'Distance', 'Personal', 'Implusive', 'Marry', 'Zodiac', 'Health', 'Problems', 'Ego', 'Family',
            'Personality', 'Gym', 'Agree', 'Movies']
x = df.loc[:, features].values
y = df.loc[:, ['Sex']].values
X = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDF, df[['Sex']]], axis=1)

"""print(finalDf)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)

targets = ['Femenino', 'Masculino']
colors = ['r', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Sex'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()"""

finalDf['Sex'] = label_encoder.fit_transform(finalDf['Sex'])

# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1, 12)).fit(finalDf)
# visualizer.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(finalDf)
print()
print("Labels:\n", kmeans.labels_)
print()
print("Inertia:\n", kmeans.inertia_)
print()
print("Number of iterations:\n", kmeans.n_iter_)
print()
print("Cluster Centers:\n", kmeans.cluster_centers_)
print()
print("How many elements in each cluster:\n", Counter(kmeans.labels_))
sns.scatterplot(data=finalDf, x='principal component 1', y='principal component 2', hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()
y = kmeans.fit_predict(principalDF)
finalDf['Cluster'] = y
print()
df1 = finalDf[finalDf.Cluster == 0]
df2 = finalDf[finalDf.Cluster == 1]
df3 = finalDf[finalDf.Cluster == 2]
print("This is cluster 1:\n", df1, "\nThis cluster has", len(df1), 'Elements\n')
print()
print("This is cluster 2:\n", df2, "\nThis cluster has", len(df2), 'Elements\n')
print()
print("This is cluster 3:\n", df3, "\nThis cluster has", len(df3), 'Elements\n')