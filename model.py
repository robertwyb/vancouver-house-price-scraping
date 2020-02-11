import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump


regions = ['Kensington-Cedar Cottage', 'Renfrew-Collingwood', 'Sunset', 'Hastings-Sunrise', 'Dunbar-Southlands',
           'Victoria-Fraserview', 'Riley Park', 'Marpole', 'Killarney', 'Kerrisdale', 'Kitsilano',
           'Grandview-Woodland', 'Arbutus-Ridge', 'Mount Pleasant', 'Shaughnessy', 'Oakridge', 'West Point Grey',
           'Fairview', 'Strathcona', 'South Cambie', 'Downtown', 'West End']
for i in regions:
    df = pd.read_csv('cleaned.csv')
    df = df.drop(columns=['CIVIC_NUMBER', 'PCOORD', 'P_PARCEL_ID', 'SITE_ID', 'Geom', 'STD_STREET', 'full_add', 'road'])
    df = df.dropna(subset=['area', 'bathroom', 'bedroom', 'built_year', 'garage', 'total_value'])
    # select only residential buildings
    df = df[df['bedroom']!=0]
    # select district
    df = df[df['Geo Local Area']==i]


    X = df[['area', 'bathroom', 'bedroom', 'built_year', 'garage']]
    y = df['total_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    model = LinearRegression(normalize=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('r2:', r2_score(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))
    dump(model, f'{i}.joblib')
