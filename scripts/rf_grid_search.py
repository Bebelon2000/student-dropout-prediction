import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


param = [
    {
        'n_estimators': [100, 200,300,400,500],
        'max_features': ["sqrt", "log2", 10],
        'max_depth': [None,10,20,30,40,50]
    },
]

gs = GridSearchCV(
    RandomForestClassifier(),
    param,
    scoring='f1',
    verbose=True
)

output_var = "Dropout"

dataset = pandas.read_csv("pre_processed/students_dropout_less_correl_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var] # actual outputs (targets)

# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)