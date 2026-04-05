import joblib
import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()

output_var = 'Dropout'

dataset = pandas.read_csv("pre_processed/students_dropout_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var] # actual outputs (targets)

test_dataset = pandas.read_csv("pre_processed/students_dropout_test.csv")
x_test = test_dataset.drop(columns=[output_var])
t_test = test_dataset[output_var] # actual outputs (targets)
# 4 ou 3
n_neighbors = 9
# p 3 weights distance
knn = KNeighborsClassifier(n_neighbors, p = 2, weights = 'uniform')
knn.fit(x_train, t_train)
joblib.dump(knn, 'models/knn_students_dropout.pkl')

y_train = knn.predict(x_train) # model predicted outputs
y_test = knn.predict(x_test)

classes = dataset[output_var].unique()

print("Training data:")
print(f"accuracy: {accuracy_score(t_train, y_train) * 100:.2f}%")

print("Testing data:")
print(f"accuracy: {accuracy_score(t_test, y_test) * 100:.2f}%")

print("Train report")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Test report")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)

display_confusion_matrix(t_train, y_train, classes, " Knn training data confusion matrix")
display_confusion_matrix(t_test, y_test, classes, " Knn test data confusion matrix")