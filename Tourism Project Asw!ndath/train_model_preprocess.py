import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
data = pd.read_csv("Travel.csv")

# Drop unwanted features
unwanted_features = ['ProdTaken', 'CustomerID', 'TypeofContact', 'DurationOfPitch', 
                     'NumberOfFollowups', 'NumberOfTrips', 'PitchSatisfactionScore','Passport', 'OwnCar']
data.drop(columns=unwanted_features, inplace=True)

# Separate target variable
X = data.drop(columns=['ProductPitched'])
y = data['ProductPitched']

# Define preprocessing steps
numeric_transformer = RobustScaler()
categorical_transformer = OneHotEncoder()
numeric_features = X.select_dtypes(include=['int', 'float']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define column transformer to apply different preprocessing to numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_preprocessed = imputer.fit_transform(X_preprocessed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict on the test data
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save preprocessor and model
dump(preprocessor, "preprocessor.joblib")
dump(knn_model, "knn_model.joblib")
