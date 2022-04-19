from flask import Flask, request
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def cupidum_api_requests():
    if request.method == 'POST':
        field = []
        loaded_minis = joblib.load('/home/H4mst3r/mysite/ML/miniKMeans.pkl')
        Religion = request.json['Religion']
        Distance = request.json['Distance']
        Personal = request.json['Personal']
        Impulsive = request.json['Impulsive']
        Marry = request.json['Marry']
        Zodiac = request.json['Zodiac']
        Health = request.json['Health']
        Problems = request.json['Problems']
        Ego = request.json['Ego']
        Family = request.json['Family']
        Personality = request.json['Personality']
        Gym = request.json['Gym']
        Agree = request.json['Agree']
        Movies = request.json['Movies']
        Sex = request.json['Sex']
        field.append(Religion)
        field.append(Distance)
        field.append(Personal)
        field.append(Impulsive)
        field.append(Marry)
        field.append(Zodiac)
        field.append(Health)
        field.append(Problems)
        field.append(Ego)
        field.append(Family)
        field.append(Personality)
        field.append(Gym)
        field.append(Agree)
        field.append(Movies)
        field.append(Sex)
        field = np.array(field).reshape(1, -1)
        df = pd.DataFrame(field, columns=['Religion', 'Distance', 'Personal', 'Impulsive', 'Marry', 'Zodiac', 'Health', 'Problems', 'Ego',
                          'Family', 'Personality', 'Gym', 'Agree', 'Movies', 'Sex'])

        import gspread
        gc = gspread.service_account(filename='/home/H4mst3r/mysite/Key/mycredentials.json')
        gsheet = gc.open_by_key("1ieph14IFEyGr2LsDJ8JmGPxreWR8MBgcCn8-7REW_HQ")
        dataTrain = pd.DataFrame(gsheet.sheet1.get_all_records())
        df1 = pd.DataFrame(dataTrain)
        del df1['Timestamp']
        del df1['Email Address']
        df1 = pd.DataFrame(df1, columns=['Religion', 'Distance', 'Personal', 'Impulsive', 'Marry', 'Zodiac', 'Health', 'Problems', 'Ego',
                           'Family', 'Personality', 'Gym', 'Agree', 'Movies', 'Sex'])
        df1 = df1.append(df, ignore_index=True)
        for (columnName, columnData) in df1.iteritems():
            df1[columnName] = df1[columnName].astype('category')
            df1[columnName] = df1[columnName].cat.codes
        features = ['Religion', 'Distance', 'Personal', 'Impulsive', 'Marry', 'Zodiac', 'Health', 'Problems', 'Ego', 'Family',
                    'Personality', 'Gym', 'Agree', 'Movies']
        # Taking values from columns and prepping them to use Principal Component Analysis
        x = df1.loc[:, features].values
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        # Adding Sex column from original DataFrame to the new PCA DataFrame
        finalDf = pd.concat([principalDF, df1[['Sex']]], axis=1)
        label_encoder = preprocessing.LabelEncoder()
        finalDf['Sex'] = label_encoder.fit_transform(finalDf['Sex'])

        y_pred = str(loaded_minis.fit_predict(df1))
        df1['Cluster'] = y_pred
        # clustNum = normDf['Cluster'].iloc[len(normDf.index) - 1]
        clustNum = df1['Cluster'].iloc[-1]
        results = clustNum[-2:-1]
        return results
