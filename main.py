import numpy as np
import pandas as pd
from pulp import *
import json
from google.cloud import storage
import os

NUTRITION_CSV = 'nutrition/nutrition.csv'

BUCKET_NAME = 'diet-planner'

NUTRITION_CSV_VARIABLE = 'TMP_NUTRITION_CSV'

TMP_NUTRITION_CSV = '/tmp/nutrition.csv'

week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
meals = ['Snack 1', 'Snack 2', 'Breakfast', 'Lunch', 'Dinner']
meal_split = {'Snack 1': 0.10, 'Snack 2': 0.10, 'Breakfast': 0.15, 'Lunch': 0.35, 'Dinner': 0.30}

data = None
split_values_meal = None
split_values_day = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def random_dataset_meal(data):
    global split_values_meal

    frac_data = data.sample(frac=1).reset_index().drop('index', axis=1)
    meal_data = []
    for s in range(len(split_values_meal) - 1):
        meal_data.append(frac_data.loc[split_values_meal[s]:split_values_meal[s + 1]])
    return dict(zip(meals, meal_data))


def random_dataset_day():
    global split_values_day

    frac_data = data.sample(frac=1).reset_index().drop('index', axis=1)
    day_data = []
    for s in range(len(split_values_day) - 1):
        day_data.append(frac_data.loc[split_values_day[s]:split_values_day[s + 1]])
    return dict(zip(week_days, day_data))


def build_nutritional_values(kg, calories):
    protein_calories = kg * 4
    carb_calories = calories / 2.
    fat_calories = calories - carb_calories - protein_calories
    res = {'Protein Calories': protein_calories, 'Carbohydrates Calories': carb_calories, 'Fat Calories': fat_calories}
    return res


def extract_gram(table):
    protein_grams = table['Protein Calories'] / 4.
    carbs_grams = table['Carbohydrates Calories'] / 4.
    fat_grams = table['Fat Calories'] / 9.
    res = {'Protein Grams': protein_grams, 'Carbohydrates Grams': carbs_grams, 'Fat Grams': fat_grams}
    return res


def model(prob, kg, calories, meal, data):
    G = extract_gram(build_nutritional_values(kg, calories))
    E = G['Carbohydrates Grams']
    F = G['Fat Grams']
    P = G['Protein Grams']
    day_data = data
    day_data = day_data[day_data.calories != 0]
    food = day_data.name.tolist()
    c = day_data.calories.tolist()
    x = pulp.LpVariable.dicts("x", indices=food, lowBound=0, upBound=1.5, cat='Continuous', indexStart=[])
    e = day_data.carbohydrate.tolist()
    f = day_data.total_fat.tolist()
    p = day_data.protein.tolist()
    #    prob  = pulp.LpProblem( "Diet", LpMinimize )
    div_meal = meal_split[meal]
    prob += pulp.lpSum([x[food[i]] * c[i] for i in range(len(food))])
    prob += pulp.lpSum([x[food[i]] * e[i] for i in range(len(x))]) >= E * div_meal
    prob += pulp.lpSum([x[food[i]] * f[i] for i in range(len(x))]) >= F * div_meal
    prob += pulp.lpSum([x[food[i]] * p[i] for i in range(len(x))]) >= P * div_meal
    prob.solve()
    variables = []
    values = []
    for v in prob.variables():
        variable = v.name
        value = v.varValue
        variables.append(variable)
        values.append(value)
    values = np.array(values).round(2).astype(float)
    sol = pd.DataFrame(np.array([food, values]).T, columns=['Food', 'Quantity'])
    sol['Quantity'] = sol.Quantity.astype(float)
    sol = sol[sol['Quantity'] != 0.0]
    sol.Quantity = (sol.Quantity * 100).astype(int)
    sol = sol.rename(columns={'Quantity': 'Quantity (g)'})
    return sol


def better_model(kg, calories):
    days_data = random_dataset_day()
    res_model = []
    for day in week_days:
        day_data = days_data[day]
        meals_data = random_dataset_meal(day_data)
        meal_model = []
        for meal in meals:
            meal_data = meals_data[meal]
            prob = pulp.LpProblem("Diet", LpMinimize)
            sol_model = model(prob, kg, calories, meal, meal_data)
            meal_model.append(sol_model.to_dict(orient='records'))
        res_model.append(meal_model)
    unpacked = []
    for i in range(len(res_model)):
        unpacked.append(dict(zip(meals, res_model[i])))
    unpacked_tot = dict(zip(week_days, unpacked))
    return unpacked_tot


def diet_internal(kg, calories):
    global data
    global split_values_meal
    global split_values_day

    if os.getenv(NUTRITION_CSV_VARIABLE) is not None:
        folder = os.getenv(NUTRITION_CSV_VARIABLE)
    else:
        folder = TMP_NUTRITION_CSV

    # Model load which only happens during cold starts
    if data is None:
        download_blob(BUCKET_NAME, NUTRITION_CSV,
                      '%s' % folder)
        data = pd.read_csv(folder).drop('Unnamed: 0', axis=1)

        data = data[['name', 'calories', 'carbohydrate', 'total_fat', 'protein']]
        data['carbohydrate'] = np.array([data['carbohydrate'].tolist()[i].split(' ') for i in range(len(data))])[:,
                               0].astype('float')
        data['protein'] = np.array([data['protein'].tolist()[i].split(' ') for i in range(len(data))])[:, 0].astype('float')
        data['total_fat'] = np.array([data['total_fat'].tolist()[i].split('g') for i in range(len(data))])[:, 0].astype(
            'float')
        split_values_day = np.linspace(0, len(data), 8).astype(int)
        split_values_day[-1] = split_values_day[-1] - 1
        split_values_meal = np.linspace(0, split_values_day[1], len(meals) + 1).astype(int)
        split_values_meal[-1] = split_values_meal[-1] - 1

        labels = []
        sizes = []

        for x, y in meal_split.items():
            labels.append(x)
            sizes.append(y)

    result = better_model(kg, calories)
    return json.dumps(result, indent=1)


def diet(request):
    print(f"Received request: {request}")
    kg = int(request.get_json().get('kg'))
    calories = int(request.get_json().get('calories'))
    print(f"Received request with kg={kg} and calories={calories}")
    result = diet_internal(kg, calories)
    print(f"Returning result: {result}")
    return result


if __name__ == '__main__':
    print(diet_internal(75, 2500))
