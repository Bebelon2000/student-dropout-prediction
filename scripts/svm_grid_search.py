import pandas
import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#c_values = np.logspace(-3, 3, 7)
#c_values = np.arange(0.00, 0.21, 0.01)
c_values = [0.1,1,10,100]

param = [
    { 'C': c_values,
      'kernel': ['linear', 'poly', 'sigmoid', 'rbf'],
      'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]

    },
]

gs = GridSearchCV(
    SVC(),
    param,
    scoring='f1',
    verbose=True
)

output_var = "Dropout"

dataset = pandas.read_csv("pre_processed/students_dropout_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var] # actual outputs (targets)

# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)