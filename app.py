import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, neighbors, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

class BirdClassifier:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_data, self.labels = self.read_images_from_folder()
        self.feature_vectors = self.extract_features()

    def read_images_from_folder(self):
        image_data = []
        labels = []
        for label in os.listdir(self.folder_path):
            label_path = os.path.join(self.folder_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.png'):
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        image_data.append(image)
                        labels.append(label)
        return image_data, labels

    def extract_features(self):
        feature_vectors = []
        for image in self.image_data:
            if image.ndim == 3:
                # Separate color channels
                channels = cv2.split(image)
            else:
                # If the image is grayscale, replicate it into three channels
                channels = [image, image, image]

            hog_features_per_channel = []
            for channel in channels:
                resized_image = cv2.resize(channel, (64, 128))
                hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                hog_features_per_channel.append(hog_features)

            # Combine HOG features from all channels
            combined_hog_features = np.hstack(hog_features_per_channel)

            feature_vectors.append(combined_hog_features)

        feature_vectors = np.vstack(feature_vectors)
        return feature_vectors

    def train_svm_classifier(self, params, cv_value):
        X_train, X_test, y_train, y_test = train_test_split(self.feature_vectors, self.labels, train_size=0.8, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough samples in the dataset.")

        classifier = svm.SVC()
        grid_search = GridSearchCV(classifier, params, cv=cv_value, n_jobs=-1,  scoring='accuracy', return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Display detailed results for each combination of hyperparameters
        results_df = pd.DataFrame(grid_search.cv_results_)
        st.write(results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        results_df.to_csv('svm_grid_search_results.csv', index=False)

        best_classifier = grid_search.best_estimator_

        # Predict on the train set
        y_train_pred = best_classifier.predict(X_train)

        # Evaluate accuracy
        accuracy_train = accuracy_score(y_train, y_train_pred)
        
        # Predict on the test set
        y_pred = best_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return best_classifier, accuracy, accuracy_train, grid_search.best_params_

    def train_knn_classifier(self, params, cv_value):
        X_train, X_test, y_train, y_test = train_test_split(self.feature_vectors, self.labels, train_size=0.8, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough samples in the dataset.")

        classifier = neighbors.KNeighborsClassifier()
        grid_search = GridSearchCV(classifier, params, cv=cv_value, n_jobs=-1, scoring='accuracy',  return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Display detailed results for each combination of hyperparameters
        results_df = pd.DataFrame(grid_search.cv_results_)
        st.write(results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        results_df.to_csv('knn_grid_search_results.csv', index=False)

        best_classifier = grid_search.best_estimator_

        # Predict on the train set
        y_train_pred = best_classifier.predict(X_train)

        # Evaluate accuracy
        accuracy_train = accuracy_score(y_train, y_train_pred)
        
        # Predict on the test set
        y_pred = best_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return best_classifier, accuracy, accuracy_train,  grid_search.best_params_

    def train_random_forest_classifier(self, params, cv_value):
        X_train, X_test, y_train, y_test = train_test_split(self.feature_vectors, self.labels, train_size=0.8, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough samples in the dataset.")

        classifier = ensemble.RandomForestClassifier()
        grid_search = GridSearchCV(classifier, params, cv=cv_value, n_jobs=-1,  scoring='accuracy', return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Display detailed results for each combination of hyperparameters
        results_df = pd.DataFrame(grid_search.cv_results_)
        st.write(results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        results_df.to_csv('random_forest_grid_search_results.csv', index=False)

        best_classifier = grid_search.best_estimator_

        # Predict on the train set
        y_train_pred = best_classifier.predict(X_train)

        # Evaluate accuracy
        accuracy_train = accuracy_score(y_train, y_train_pred)
        
        # Predict on the test set
        y_pred = best_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return best_classifier, accuracy, accuracy_train, grid_search.best_params_
    
    def train_neural_network_classifier(self, params, cv_value):
        X_train, X_test, y_train, y_test = train_test_split(self.feature_vectors, self.labels, train_size=0.8, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough samples in the dataset.")

        classifier = MLPClassifier()
        grid_search = GridSearchCV(classifier, params, cv=cv_value, n_jobs=-1, scoring='accuracy', return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Display detailed results for each combination of hyperparameters
        results_df = pd.DataFrame(grid_search.cv_results_)
        st.write(results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        results_df.to_csv('neural_network_grid_search_results.csv', index=False)

        best_classifier = grid_search.best_estimator_

        # Predict on the train set
        y_train_pred = best_classifier.predict(X_train)

        # Evaluate accuracy on the train set
        accuracy_train = accuracy_score(y_train, y_train_pred)

        # Predict on the test set
        y_pred = best_classifier.predict(X_test)

        # Evaluate accuracy on the test set
        accuracy = accuracy_score(y_test, y_pred)

        return best_classifier, accuracy, accuracy_train, grid_search.best_params_

def main():
    folder_path = r'D:/VIXY/Penambangan Data/Dataset/training_fish'
    bird_classifier = BirdClassifier(folder_path)

    # List untuk menyimpan hasil percobaan
    results = []

    # Choose a classification method
    st.sidebar.title("Fish Classification App")
    choice = st.sidebar.selectbox("Choose a classification method:", ["SVM Classifier", "K-Nearest Neighbors (KNN) Classifier", "Random Forest Classifier", "Neural Network"])

    cv_value = 2
    if choice == "SVM Classifier":
        method = 'svm'
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        trained_classifier, accuracy, accuracy_train, best_params = bird_classifier.train_svm_classifier(params, cv_value)
    elif choice == "K-Nearest Neighbors (KNN) Classifier":
        method = 'knn'
        params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        trained_classifier, accuracy, accuracy_train, best_params = bird_classifier.train_knn_classifier(params, cv_value)
    elif choice == "Random Forest Classifier":
        method = 'random_forest'
        params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        trained_classifier, accuracy, accuracy_train, best_params = bird_classifier.train_random_forest_classifier(params, cv_value)
    elif choice == "Neural Network":
        method = 'neural_network'
        params = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}
        trained_classifier, accuracy, accuracy_train, best_params = bird_classifier.train_neural_network_classifier(params, cv_value)
    else:
        st.sidebar.error("Invalid choice. Please choose a valid classification method.")
        return

    results.append({'Method': method.capitalize(), 'Best Parameters': best_params, 'Accuracy Training': accuracy_train, 'Accuracy Testing': accuracy})

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file with a specified directory and filename
    results_df.to_csv(f'classification_results_{method}.csv', index=False)

    st.subheader(f"{method.capitalize()} Classifier:")
    st.write(f"Best Parameters: {best_params}")
    st.write(f"Accuracy Testing: {accuracy}")
    st.write(f"Accuracy Training: {accuracy_train}")

if __name__ == '__main__':
    main()
