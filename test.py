import pandas as pd
import joblib
import random

model = joblib.load('decision_tree.joblib')


distance_from_home = []
distance_from_last_transaction = []
ratio_to_median_purchase_price = []
repeat_retailer = []
used_chip = []
used_pin_number = []
online_order = []
sample_transaction = []
test_data = []

for i in range(20):
    # generate random data for testing
    distance_from_home = random.randint(1, 100)
    distance_from_last_transaction = random.randint(1, 500) / 10
    ratio_to_median_purchase_price = random.randint(1, 50) / 10
    repeat_retailer = random.choice((0, 1))
    used_chip = random.choice((0, 1))
    used_pin_number = random.choice((0, 1))
    online_order = random.choice((0, 1))
    test_data.append(
        [distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
         repeat_retailer, used_chip, used_pin_number, online_order])

prediction = model.predict(test_data)

prediction_no = 0

for i in test_data:
    if int(prediction[prediction_no]):
        print("*** FRAUD ***")
    print(f"""
distance_from_home = {i[0]}
distance_from_last_transaction = {i[1]}
ratio_to_median_purchase_price = {i[2]}
repeat_retailer = {i[3]}
used_chip = {i[4]}
used_pin_number = {i[5]}
online_order = {i[6]}
fraud = {int(prediction[prediction_no])}
    """)
    prediction_no += 1
