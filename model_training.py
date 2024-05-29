import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the data
df = pd.read_csv('heart.csv')

# Split the data into features (X) and target (y)
X = df.drop(columns=['output'])
y = df['output']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')

# Save the trained model to a file
with open('rf_clf.pkl', 'wb') as file:
    pickle.dump(clf, file)
