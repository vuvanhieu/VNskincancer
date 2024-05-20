import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_fscore_support, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
import joblib
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define the experiment folder
experiment_folder = "/data2/cmdir/home/anhnv/vvhieu/Hien/Multimedia_Tools_and_Applications_SkinLesion_BCC_SCC_MM_(before_Review)"
os.makedirs(experiment_folder, exist_ok=True)

model_name = 'stack_SMOTE_Validation_Fold_threeCNN_FC2_VGG16_BCC_SCC_MM_10'
result_folder = os.path.join(experiment_folder, model_name)
os.makedirs(result_folder, exist_ok=True)

# Set the paths to the train and test folders
train_folder = "/data2/cmdir/home/anhnv/vvhieu/Hien/Hien_Data_Feature/train_Hien_VGG19_FC_2_full"
test_folder = "/data2/cmdir/home/anhnv/vvhieu/Hien/Hien_Data_Feature/test_Hien_VGG19_FC_2_full"

# Define the categories
categories = ['BCC', 'MM', 'SCC']
num_categories = len(categories)

# Function to load .npy files from a folder
def load_npy_files(folder, categories):
    images = []
    labels = []
    file_paths = []
    class_folders = sorted(os.listdir(folder))
    for class_folder in class_folders:
        if class_folder in categories:  # Only process folders that are in the specified categories
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.npy'):
                        img_path = os.path.join(class_path, filename)
                        try:
                            img = np.load(img_path)
                            if img is not None:
                                images.append(img)
                                labels.append(class_folder)
                                file_paths.append(img_path)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels), file_paths

# Function to get the shape of each category in a folder
def get_category_shapes(folder, categories):
    category_shapes = []
    for category in categories:
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            image_files = [f for f in os.listdir(category_path) if f.endswith('.npy')]
            num_images = len(image_files)
            category_shapes.append({'Category': category, 'Number of Images': num_images})
    return category_shapes

# Load .npy files for train and test datasets
train_images, train_labels, _ = load_npy_files(train_folder, categories)
test_images, test_labels, test_file_paths = load_npy_files(test_folder, categories)

# Debugging: Check if images are loaded correctly
print(f'Loaded {len(train_images)} training images.')
print(f'Loaded {len(test_images)} testing images.')

# Check if images were loaded successfully
if len(train_images) == 0 or len(test_images) == 0:
    raise ValueError("No images were loaded. Check the file paths and formats of the images in the folders.")

# Get the shape of each category in the train and test folders
train_category_shapes = get_category_shapes(train_folder, categories)
test_category_shapes = get_category_shapes(test_folder, categories)

# Save the shapes to a CSV file
category_shapes_df = pd.DataFrame(train_category_shapes + test_category_shapes)
category_shapes_csv_path = os.path.join(result_folder, 'category_shapes.csv')
category_shapes_df.to_csv(category_shapes_csv_path, index=False)

print('Category shapes saved to:', category_shapes_csv_path)

# Shuffle the train data
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

# Split test data to use 50% for validation during training
test_images, val_images, test_labels, val_labels, test_file_paths, val_file_paths = train_test_split(
    test_images, test_labels, test_file_paths, test_size=0.5, random_state=42)

# Flatten images for fully connected neural network
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
val_images_flattened = val_images.reshape(val_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Debugging: Check label distribution
print("Train label distribution:", np.bincount(train_labels_encoded))
print("Validation label distribution:", np.bincount(val_labels_encoded))
print("Test label distribution:", np.bincount(test_labels_encoded))

# Debugging: Check label distribution before SMOTE
print("Train label distribution before SMOTE:", np.bincount(train_labels_encoded))
print("Shape of train images before SMOTE:", train_images_flattened.shape)

# Use SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
train_images_flattened_resampled, train_labels_encoded_resampled = smote.fit_resample(train_images_flattened, train_labels_encoded)


# Debugging: Check label distribution after SMOTE
print("Train label distribution after SMOTE:", np.bincount(train_labels_encoded_resampled))
print("Shape of train images after SMOTE:", train_images_flattened_resampled.shape)

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels_encoded_resampled, num_categories)
val_labels = to_categorical(val_labels_encoded, num_categories)
test_labels = to_categorical(test_labels_encoded, num_categories)

# Debugging: Check shape of one-hot encoded labels
print("Shape of train labels:", train_labels.shape)
print("Shape of validation labels:", val_labels.shape)
print("Shape of test labels:", test_labels.shape)

# Create three different Keras models
def create_keras_model_3_1(X_train, y_train):
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_keras_model_3_2(X_train, y_train):
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_keras_model_3_3(X_train, y_train):
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

epoch_values = [3, 10, 20, 40, 60, 80, 100]
# epoch_values = [2, 3]
metrics_list = []
validation_metrics = []

# Define functions for classification report and confusion matrix
def classification_report_csv(report, result_out, model_name):
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(result_out, model_name + '_classification_report.csv')
    report_df.to_csv(report_csv_path)
    print('Classification report saved to:', report_csv_path)

def confusion_matrix_report(cm, result_out, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + '_confusion_matrix.png'))
    plt.close()

def accuracy_loss_plot(result_out, epoch, history, model_name):
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, model_name + '_accuracy_plot.png'))
    plt.close()

    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, model_name + '_loss_plot.png'))
    plt.close()

def load_all_models(best_model_paths):
    all_models = []
    for model_path in best_model_paths:
        model = load_model(model_path)
        all_models.append(model)
        print(f'>loaded {model_path}')
    return all_models

def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        yhat = model.predict(inputX, verbose=0)
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def fit_stacked_model_with_cv(members, inputX, inputy, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    stacked_models = []
    cv_scores = []

    for train_index, test_index in kf.split(inputX):
        trainX, testX = inputX[train_index], inputX[test_index]
        trainy, testy = inputy[train_index], inputy[test_index]
        
        # Create stacked dataset for training
        stacked_trainX = stacked_dataset(members, trainX)
        # Create stacked dataset for validation
        stacked_testX = stacked_dataset(members, testX)

        # Fit the meta-model (Logistic Regression in this case)
        model = LogisticRegression(max_iter=1000)
        model.fit(stacked_trainX, trainy)
        
        # Save the trained model
        stacked_models.append(model)
        
        # Evaluate the model on the validation fold
        yhat = model.predict(stacked_testX)
        score = np.mean(yhat == testy)
        cv_scores.append(score)
        print(f'Fold score: {score}')

    print(f'Cross-validated score: {np.mean(cv_scores)}')
    
    return stacked_models, np.mean(cv_scores)

def stacked_prediction(members, models, inputX):
    stackedX = stacked_dataset(members, inputX)
    # Get class probabilities from each model
    predictions = np.array([model.predict_proba(stackedX) for model in models])
    # Average the probabilities
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

def save_validation_metrics(validation_metrics, result_folder):
    validation_metrics_df = pd.DataFrame(validation_metrics)
    validation_metrics_file = os.path.join(result_folder, 'validation_metrics.csv')
    validation_metrics_df.to_csv(validation_metrics_file, index=False)
    return validation_metrics_file

def select_best_models(validation_metrics_file):
    metrics_df = pd.read_csv(validation_metrics_file)
    sorted_metrics_df = metrics_df.sort_values(by='Validation Accuracy', ascending=False)
    best_models = []
    selected_models = set()
    
    for _, row in sorted_metrics_df.iterrows():
        model_name = row['Model Name']
        if model_name not in selected_models and len(best_models) < 3:
            best_models.append(os.path.join(result_folder, f'epoch_{row["Epoch"]}', f'{model_name}.h5'))
            selected_models.add(model_name)
    # Save the best models info
    best_models_info = sorted_metrics_df.head(3)
    best_models_info.to_csv(os.path.join(result_folder, 'best_models_info.csv'), index=False)
    return best_models

# Train multiple models for each epoch value
for epoch in epoch_values:
    result_out = os.path.join(result_folder, 'epoch_' + str(epoch))
    os.makedirs(result_out, exist_ok=True)

    # Train and save multiple models
    for i in range(3):
        if i == 0:
            model = create_keras_model_3_1(train_images_flattened_resampled, train_labels)
        elif i == 1:
            model = create_keras_model_3_2(train_images_flattened_resampled, train_labels)
        else:
            model = create_keras_model_3_3(train_images_flattened_resampled, train_labels)

        history = model.fit(train_images_flattened_resampled, train_labels, epochs=epoch, batch_size=64,
                            validation_data=(val_images_flattened, val_labels), verbose=2)
        model.save(os.path.join(result_out, f'model_{i}.h5'))
        
        # Generate and save classification report and confusion matrix
        predictions = model.predict(test_images_flattened)
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_labels = label_encoder.inverse_transform(np.argmax(test_labels, axis=1))
        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
        report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True, zero_division=1)
        classification_report_csv(report, result_out, f'model_{i}')
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        confusion_matrix_report(cm, result_out, f'model_{i}')
        
        # Plot accuracy and loss
        accuracy_loss_plot(result_out, epoch, history, f'model_{i}')
        
        # Calculate and save metrics for the model
        accuracy = np.mean(y_pred_classes == np.argmax(test_labels, axis=1))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='macro', zero_division=1)
        sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
        specificity = np.mean([cm[i, i] / (np.sum(cm[:, i]) or 1) for i in range(len(cm))])
        auc_score = roc_auc_score(test_labels, predictions, average='macro')
        loss = history.history['loss'][-1]
        time_taken = (history.epoch[-1] + 1) * history.params['steps']

        metrics = {
            'Model Name': f'model_{i}',
            'Batch Size': 64,
            'Epoch': epoch,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'AUC Score': auc_score,
            'Loss': loss,
            'Epoch Training Time': time_taken,
            'Validation Accuracy': history.history['val_accuracy'][-1]
        }
        metrics_list.append(metrics)
        
        # Save validation metrics
        validation_metrics.append({
            'Model Name': f'model_{i}',
            'Epoch': epoch,
            'Batch Size': 64,
            'Validation Accuracy': history.history['val_accuracy'][-1]
        })

# Save validation metrics to CSV
validation_metrics_file = save_validation_metrics(validation_metrics, result_folder)

# Select the best models based on validation accuracy
best_model_paths = select_best_models(validation_metrics_file)

# Load the best models for stacking
best_models = load_all_models(best_model_paths)

# Fit stacked model using cross-validation
start_time = datetime.now()
stacked_models, cv_score = fit_stacked_model_with_cv(best_models, val_images_flattened, np.argmax(val_labels, axis=1))
end_time = datetime.now()
time_taken_stack = end_time - start_time

# Save the stacked models
for i, model in enumerate(stacked_models):
    joblib.dump(model, os.path.join(result_folder, f'stacked_model_{i}.sav'))

# Evaluate stacked model on test data
stacked_predictions = stacked_prediction(best_models, stacked_models, test_images_flattened)

# Debugging: Print shape of stacked_predictions
print(f'Shape of stacked_predictions: {stacked_predictions.shape}')

# Check if stacked_predictions has the correct shape
if len(stacked_predictions.shape) == 1:
    print(f"Error: stacked_predictions has shape {stacked_predictions.shape}")
else:
    stacked_yhat = np.argmax(stacked_predictions, axis=1)
    stacked_accuracy = np.mean(stacked_yhat == np.argmax(test_labels, axis=1))
    print(f'Accuracy of stacked model with cross-validation: {stacked_accuracy:.4f}')

    # Ensure predictions are within the expected label range
    stacked_yhat = np.clip(stacked_yhat, 0, len(label_encoder.classes_) - 1)

    # Generate and save classification report for stacked model
    y_true_labels = label_encoder.inverse_transform(np.argmax(test_labels, axis=1))  # Update to get correct labels
    y_pred_labels = label_encoder.inverse_transform(stacked_yhat)
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True, zero_division=1)
    classification_report_csv(report, result_folder, 'stacked_model')

    # Calculate and save metrics for stacked model
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='macro', zero_division=1)
    sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    specificity = np.mean([cm[i, i] / (np.sum(cm[:, i]) or 1) for i in range(len(cm))])

    # Ensure the stacked_predictions are in the correct shape for AUC
    if stacked_predictions.shape[1] == num_categories:
        auc_score = roc_auc_score(test_labels, stacked_predictions, average='macro')
        loss = log_loss(test_labels, stacked_predictions)
    else:
        auc_score = 'N/A'
        loss = 'N/A'

    metrics = {
        'Model Name': 'stacked_model',
        'Batch Size': 64,
        'Epoch': epoch,
        'Accuracy': stacked_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC Score': auc_score,
        'Loss': loss,
        'Epoch Training Time': time_taken_stack.total_seconds()
    }
    metrics_list.append(metrics)

    # Plot confusion matrix for stacked model
    confusion_matrix_report(cm, result_folder, 'stacked_model')

    # Plot ROC curve for each class in the stacked model
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_categories):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], stacked_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'red', 'green']
    for i in range(num_categories):
        plt.plot(fpr[i], tpr[i], color=colors[i], label=f'ROC curve ({categories[i]}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_folder, f'{model_name}_roc_curve.png'))
    plt.close()
    print('ROC curve saved to:', os.path.join(result_folder, f'{model_name}_roc_curve.png'))

    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(result_folder, f'{model_name}_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    print('Metrics summary saved to:', metrics_csv_path)

    # Make predictions using the stacked model
    stacked_yhat = np.argmax(stacked_predictions, axis=1)

    # Ensure predictions are within the expected label range
    stacked_yhat = np.clip(stacked_yhat, 0, len(label_encoder.classes_) - 1)

    # Get the true labels for the test set
    true_labels_str = label_encoder.inverse_transform(np.argmax(test_labels, axis=1))
    predicted_labels_str = label_encoder.inverse_transform(stacked_yhat)

    # Debugging: Print lengths of arrays
    print(f'Length of test_file_paths: {len(test_file_paths)}')
    print(f'Length of true_labels_str: {len(true_labels_str)}')
    print(f'Length of predicted_labels_str: {len(predicted_labels_str)}')

    # Save predictions to a CSV file
    if len(test_file_paths) == len(true_labels_str) == len(predicted_labels_str):
        predictions_df = pd.DataFrame({
            'File Path': test_file_paths,
            'True Label': true_labels_str,
            'Predicted Label': predicted_labels_str
        })
        predictions_csv_path = os.path.join(result_folder, 'predictions_on_test_set.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        print('Predictions on test set saved to:', predictions_csv_path)
    else:
        print('Error: Mismatch in array lengths. Check the data for inconsistencies.')

    # Generate and save classification report and confusion matrix
    report = classification_report(true_labels_str, predicted_labels_str, target_names=categories, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(result_folder, 'stacked_model_classification_report.csv')
    report_df.to_csv(report_csv_path)
    print('Classification report saved to:', report_csv_path)

    cm = confusion_matrix(true_labels_str, predicted_labels_str)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_folder, f'{model_name}_confusion_matrix.png'))
    plt.close()
    print('Confusion matrix saved to:', os.path.join(result_folder, f'{model_name}_confusion_matrix.png'))

    # Plot ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    stacked_predictions = stacked_dataset(best_models, test_images_flattened)

    for i in range(num_categories):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], stacked_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'red', 'green']
    for i in range(num_categories):
        plt.plot(fpr[i], tpr[i], color=colors[i], label=f'ROC curve ({categories[i]}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_folder, f'{model_name}_roc_curve.png'))
    plt.close()
    print('ROC curve saved to:', os.path.join(result_folder, f'{model_name}_roc_curve.png'))
