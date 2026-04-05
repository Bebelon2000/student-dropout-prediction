import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param = [
    { 'n_neighbors': range(1,21),
      'weights': ['uniform', 'distance'],
      # p=1 para Manhattan, p=2 para Euclidiana
      'p': [1, 2,] },
]

gs = GridSearchCV(
    KNeighborsClassifier(),
    param,
    #Recall f1
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