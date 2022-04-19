import gspread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
import pandas as pd
#######################################
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
#######################################
import joblib
import json

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

gc = gspread.service_account(filename='mycredentials.json')
gsheet = gc.open_by_key("1ieph14IFEyGr2LsDJ8JmGPxreWR8MBgcCn8-7REW_HQ")
# values_list = gsheet.sheet1.row_values(83)
# print(values_list)
df = pd.DataFrame(gsheet.sheet1.get_all_records())
del df['Timestamp']
########################################################################################################################
#  Changing string elements to numbers so the classifier can work
for (columnName, columnData) in df.iteritems():
    df[columnName] = df[columnName].astype('category')
    df[columnName] = df[columnName].cat.codes
scaler = MinMaxScaler()
normDf = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
# Dropping column with email address since this column isn't needed
del normDf['Email Address']
# Renaming columns from phrases to string
normDf = normDf.rename(columns={'¿Qué tan religios@ eres?': 'Religion',
                                '¿Crees que las relaciones a distancia pueden funcionar?': 'Distance',
                                '¿Hablas de tus problemas personales con tus amigos cercanos?': 'Personal',
                                '¿Eres propenso a tomar decisiones impulsivas?': 'Impulsive',
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
# print(normDf)
########################################################################################################################
features = ['Religion', 'Distance', 'Personal', 'Impulsive', 'Marry', 'Zodiac', 'Health', 'Problems', 'Ego', 'Family',
            'Personality', 'Gym', 'Agree', 'Movies']
# Taking values from columns and prepping them to use Principal Component Analysis
x = normDf.loc[:, features].values
y = normDf.loc[:, ['Sex']].values
X = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
# Creating DataFrame from the newly generated data from PCA
principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# Adding Sex column from original DataFrame to the new PCA DataFrame
finalDf = pd.concat([principalDF, normDf[['Sex']]], axis=1)
# Changing Sex from string to number using label encoder
label_encoder = preprocessing.LabelEncoder()
finalDf['Sex'] = label_encoder.fit_transform(finalDf['Sex'])
x_fin = normDf.loc[:, features].values
y_fin = normDf.loc[:, ['Sex']].values
X_train, X_test, y_train, y_test = train_test_split(x_fin, y_fin, test_size=0.3, random_state=0)
# print(finalDf)
# print()
# print('\n', finalDf.iloc[len(finalDf.index) - 1:, :])
# print()

# Initializing the MiniKMeans model to be used
mini = MiniBatchKMeans(n_clusters=3)
# mini.fit(finalDf)
minMod = mini.fit(X_train, y_train)
joblib.dump(minMod, 'miniKMeans.pkl')
carryMod = joblib.load('miniKMeans.pkl')


# Predicting the cluster of each row of data passed to the model and adding column with this information
y1 = minMod.fit_predict(principalDF)
finalDf['Cluster'] = y1
clustNum = finalDf['Cluster'].iloc[len(finalDf.index) - 1]
print("This is the returned result:", clustNum)
print(normDf.columns)
fin_mod = finalDf.to_json()
print(fin_mod)
print(type(clustNum))

print(finalDf)
print()
print("Inertia:\n", mini.inertia_)
print()
print("Number of iterations:\n", mini.n_iter_)
print()
print("Cluster Centers:\n", mini.cluster_centers_)
print()
print("How many elements in each cluster:\n", Counter(mini.labels_))

sns.scatterplot(data=finalDf, x='principal component 1', y='principal component 2', hue=mini.labels_)
plt.scatter(mini.cluster_centers_[:, 0], mini.cluster_centers_[:, 1],
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()
