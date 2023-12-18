import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.tree import export_graphviz

df = pd.read_csv('card_transdata.csv')  # creating a dataframe

data_input = df.drop(columns=['fraud'])
data_output = df['fraud']

model = DecisionTreeClassifier()
model.fit(data_input, data_output)
joblib.dump(model, 'decision_tree.joblib')

# # Measure accuracy
# input_train, input_test, output_train, output_test = train_test_split(data_input, data_output, test_size=0.2)
# model = DecisionTreeClassifier()
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print(score)

export_graphviz(model, out_file='decision_tree_viz.dot',
                feature_names=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
                               'repeat_retailer', 'used_chip', 'used_pin', 'online_order'],
                class_names=['fraud','not fraud'],
                label='all', rounded=True, filled=True)
