import gspread
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from matplotlib import pyplot as plt

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
df['Sexo'] = label_encoder.fit_transform(df['Sexo'])
df['Email Address'] = label_encoder.fit_transform(df['Email Address'])
Y_var = df['Sexo'].values
X_var = df[['¿Qué tan religios@ eres?',
            '¿Crees que las relaciones a distancia pueden funcionar?',
            '¿Hablas de tus problemas personales con tus amigos cercanos?',
            '¿Eres propenso a tomar decisiones impulsivas?', '¿Quieres casarte?',
            '¿Crees en los horóscopos y signos del zodiaco?',
            '¿Qué tan consciente eres de tu salud?',
            '¿Te gusta hablar sobre tus problemas?',
            '¿Tu ego te impide disculparte cuando te equivocas?',
            '¿Cómo resuelves los problemas con tus amigos/familiares? [Respuesta:]',
            '¿Con qué tipo de personalidad te identificas? [Respuesta:]',
            '¿Haces ejercicio?',
            'Si no estoy de acuerdo contigo en algunos temas, ¿cómo te sientes? [Respuesta:]',
            '¿Prefieres ver películas o leer libros? [Respuesta:]',
            'Sexo']].values
wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=0)
    k_means.fit(X_var)
    wcss.append(k_means.inertia_)
plt.plot(np.arange(1, 11), wcss)
plt.xlabel('Clusters')
plt.ylabel('SSE')
# plt.show()

k_means_optimum = KMeans(n_clusters=6, init='k-means++', random_state=0)
y = k_means_optimum.fit_predict(X_var)
# print(y)
df['cluster'] = y
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
# print()
# print("This is the whole dataset:", df, '\n')
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
df6 = df[df.cluster == 5]
# print()
print("This is cluster 1:\n", df1, "\nThis cluster has", len(df1), 'Elements\n')
print()
print("This is cluster 2:\n", df2, "\nThis cluster has", len(df2), 'Elements\n')
print()
print("This is cluster 3:\n", df3, "\nThis cluster has", len(df3), 'Elements\n')
print()
print("This is cluster 4:\n", df4, "\nThis cluster has", len(df4), 'Elements\n')
print()
print("This is cluster 5:\n", df5, "\nThis cluster has", len(df5), 'Elements\n')
print()
print("This is cluster 6:\n", df6, "\nThis cluster has", len(df6), 'Elements\n')
