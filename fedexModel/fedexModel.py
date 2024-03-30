import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import networkx as nx

def predict_delivery_time(csv_file_path, weather_condition, delivery_area, num_packages): 
    data = pd.read_csv(csv_file_path)
    predictions = {}
    unique_drivers = data['Driver ID'].unique()
    for driver_id in unique_drivers:
        driver_data = data[data['Driver ID'] == driver_id]
        
        categorical_features = ['Weather Conditions', 'Delivery Area']
        numerical_features = ['Number of Packages']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ])
        
        model = RandomForestRegressor()
        
        X = driver_data.drop(columns=['Time Taken (minutes)', 'Driver ID'])
        y = driver_data['Time Taken (minutes)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        
        input_data = pd.DataFrame({
            'Weather Conditions': [weather_condition],
            'Delivery Area': [delivery_area],
            'Number of Packages': [num_packages]
        })
        
        prediction = pipeline.predict(input_data)
        
        predictions[driver_id] = prediction[0]
    
    return predictions

csv_file_path = 'fedex_data.csv'
predictions = predict_delivery_time(csv_file_path, 'Cloudy', 'Rural', 100)
for driver_id, prediction in predictions.items():
    print("Predicted time for delivery for Driver", driver_id, ":", prediction, "minutes")




def assign_routes_to_drivers(driver_data_csv, routes_csv):
    driver_data = pd.read_csv(driver_data_csv)
    routes_data = pd.read_csv(routes_csv)
    drivers = driver_data['Driver ID'].unique()
    routes = [(route['Route Number'], route['Weather Conditions'], route['Delivery Area'], route['Number of Packages']) for idx, route in routes_data.iterrows()]
    
    G = nx.Graph()
    
    for route in routes:
        G.add_node(route)
    
    for route1 in routes:
        for route2 in routes:
            for driver in drivers:
                prediction = predict_delivery_time(driver_data_csv, route2[1], route2[2], route2[3])
                weight = prediction[driver]
                G.add_edge(route1, route2, weight=weight)
    
    start_node = (0, '', '', 0)
    G.add_node(start_node)
    
    for route in routes:
        prediction = predict_delivery_time(driver_data_csv, route[1], route[2], route[3])
        weight = prediction[driver]
        G.add_edge(start_node, route, weight=weight)
    
    #dijkstra's
    assigned_routes = {}
    for driver in drivers:
        shortest_path = nx.single_source_dijkstra_path(G, start_node, weight='weight')
        if len(shortest_path) > 1: 
            assigned_routes[driver] = shortest_path[-1]
        else:
            assigned_routes[driver] = None 
    
    return assigned_routes

driver_data_csv = 'fedex_data.csv'
routes_csv = 'routes.csv'

#assigned_routes = assign_routes_to_drivers(driver_data_csv, routes_csv)
#for driver, route in assigned_routes.items():
#    print(f"Driver {driver} assigned to route {route}")
