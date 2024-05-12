import os
import numpy as np
import pandas as pd
import time
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle
from pprint import pprint
import joblib

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from numpy import dstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ====================================================================
import os
directory_work = os.getcwd()
experient_folder = os.path.join(directory_work, 'experient_folder')
model_name = 'experient1_Stack_vgg16_features'

result_folder = os.path.join(experient_folder, model_name)
os.makedirs(result_folder, exist_ok=True)

# Define the categories
categories = ['BCC', 'MM', 'SCC', 'no skin cancer']
num_categories = len(categories)

# Define a list of epoch values to try
batch_sizes = [16,32]
epoch_values = [10, 14, 20, 40, 60, 80, 100]

# epoch_values = [1]
# batch_sizes = [16]


train_folder = os.path.join(directory_work, 'train_Deep_Features/vgg16_features')
test_folder = os.path.join(directory_work, 'test_Deep_Features/vgg16_features')


# # ==================================================================
import pandas as pd

def save_model_configurations_to_csv(input_dim, num_classes, result_out):
    models = {
        "Model 1": create_keras_model_1(input_dim, num_classes),
        "Model 2": create_keras_model_2(input_dim, num_classes),
        "Model 3": create_keras_model_3(input_dim, num_classes)
    }

    model_details = []
    for name, model in models.items():
        for layer in model.layers:
            config = layer.get_config()
            model_details.append({
                "Model": name,
                "Layer": layer.__class__.__name__,
                "Units": config.get("units", "N/A"),
                "Activation": config.get("activation", "N/A"),
                "Dropout Rate": config.get("rate", "N/A")
            })

    df = pd.DataFrame(model_details)
    df.to_csv(os.path.join(result_out, 'model_configurations.csv'), index=False)
    print("Model configurations saved to 'model_configurations.csv'.")

# # ==================================================================
def save_to_csv(inputX, inputy, result_out):
    print(f"inputy shape: {inputy.shape}")  # This will show you the shape of inputy
    # Check if inputy is a 2D array with one column; if not, reshape or adjust accordingly
    if len(inputy.shape) > 1 and inputy.shape[1] != 1:
        inputy = inputy.argmax(axis=1)  # Assuming inputy is one-hot encoded, convert it to label encoding

    df = pd.DataFrame(inputX)
    df['Label'] = inputy  # Now inputy should correctly fit as a single column
    
    csv_filename = os.path.join(result_out, 'stacked_input_data.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")
# # ==================================================================

from sklearn.metrics import log_loss

def collect_stacked_model_metrics(members, stacked_model, X_test, y_test, result_out, model_name, batch_size, epoch, training_time):
    # Predict using the stacked model
    y_pred_proba = stacked_prediction(members, stacked_model, X_test)  # get probability estimates
    y_pred_multiclass = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels

    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred_multiclass)
    precision = precision_score(y_test, y_pred_multiclass, average='weighted')
    recall = recall_score(y_test, y_pred_multiclass, average='weighted')
    f1 = f1_score(y_test, y_pred_multiclass, average='weighted')
    auc_score = roc_auc_score(to_categorical(y_test, num_categories), y_pred_proba, average='macro', multi_class='ovr')

    # Calculate log loss, which can serve as a loss metric
    logloss = log_loss(y_test, y_pred_proba)

    # Sensitivity and specificity (calculated per class)
    cm = confusion_matrix(y_test, y_pred_multiclass)
    sensitivity = {}
    specificity = {}
    class_labels = label_encoder.classes_
    for i, label in enumerate(class_labels):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        sensitivity[label] = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity[label] = TN / (TN + FP) if (TN + FP) != 0 else 0

    # Combine all metrics in a dictionary    
    metrics = {'Model Name': model_name,
               'batch_sizes': batch_size,
               'Epoch': epoch, 
               'accuracy': accuracy, 
               'precision': precision,
               'recall': recall, 
               'f1':f1,
               'Sensitivity': sensitivity,
               'Specificity': specificity,
               'auc_scores': auc_score,
               'epoch_training_time': training_time}
        
    return metrics

# # ==================================================================
def collect_metrics(model, X_test, y_test, result_out, model_name, epoch, batch_size, training_time):
    # Predict the test dataset
    predictions = model.predict(X_test)
    predictions_multiclass = np.argmax(predictions, axis=1)

    # Basic metrics
    accuracy = accuracy_score(y_test, predictions_multiclass)
    precision = precision_score(y_test, predictions_multiclass, average='weighted')
    recall = recall_score(y_test, predictions_multiclass, average='weighted')
    f1 = f1_score(y_test, predictions_multiclass, average='weighted')
    auc_score = roc_auc_score(to_categorical(y_test, num_classes=num_categories), predictions, average='macro')

    # Confusion matrix for sensitivity and specificity
    cm = confusion_matrix(y_test, predictions_multiclass)
    sensitivity = {}
    specificity = {}
    class_labels = label_encoder.classes_

    # Calculate sensitivity and specificity for each class
    for i, label in enumerate(class_labels):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        sensitivity[label] = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity[label] = TN / (TN + FP) if (TN + FP) != 0 else 0

    # Combine all metrics in a dictionary
    metrics = {'Model Name': model_name,
               'batch_sizes': batch_size,
               'Epoch': epoch, 
               'accuracy': accuracy, 
               'precision': precision,
               'recall': recall, 
               'f1':f1,
               'Sensitivity': sensitivity,
               'Specificity': specificity,
               'auc_scores': auc_score,
               'epoch_training_time': training_time}
    
    return metrics

# # ==================================================================
def accuracy_loss_plot(result_out, epoch, history, model_name):
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, str(model_name) +'_accuracy_plot.png'))
    # plt.show()
    plt.close()

    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, str(model_name) +'_loss_plot.png'))
#     plt.show()
    plt.close()  
    
#==================================================
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_stack_roc_curves(y_true, y_pred, result_out, model_name, class_labels):
    # Number of classes
    n_classes = y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Multi-class ROC for ' + model_name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_out, model_name + 'stack_roc_curves.png'))
    plt.close()
    
# # ================================================================== 
def confusion_matrix_report(model, test_images_normalized, test_labels_encoded, result_out, model_name):
    # Make predictions on the test set
    y_pred = model.predict(test_images_normalized)
    # Convert y_pred to multiclass format
    y_pred_multiclass = np.argmax(y_pred, axis=1)
    # Generate the confusion matrix for the stacked model
    confusion_matrix_stacked = confusion_matrix(test_labels_encoded, y_pred_multiclass) 
    # Get the class labels
    class_labels = label_encoder.classes_
    
    # Plot the confusion matrix with class labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_stacked, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
    plt.savefig(os.path.join(result_out, str(model_name)+'_confusion_matrix.png'))
    # plt.show()
    plt.close()
        
    test_labels_multilabel = to_categorical(test_labels_encoded, num_classes=num_categories)


    # Calculate the false positive rate (FPR) and true positive rate (TPR) for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(test_labels_multilabel[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(len(class_labels)):
        linestyle = '--' if i % 2 == 0 else '-'
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})', linestyle=linestyle)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_out, str(model_name)+'_roc_curve.png'))
    # plt.show()
    plt.close()
        
# # ==================================================================    
# Define the wrapper functions for Keras models
def create_keras_model_1(input_dim, num_classes):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_keras_model_2(input_dim, num_classes):
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# # ==================================================================
def create_keras_model_3(input_dim, num_classes):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ============================================================================
# load models from file
def load_all_models(n_models, result_out):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = os.path.join(result_out,  f'model_{i}.h5')
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
# # ==================================================================

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# # ==================================================================
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    
    # fit standalone model
    model = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors as required
    model.fit(stackedX, inputy)
    
    # Now you can print the expected feature shape as the model is defined and fitted
    print(f"Stacked feature shape: {stackedX.shape}")
    print(f"Expected feature shape by KNeighborsClassifier: {model.n_features_in_}")
    
    return model

# # ==================================================================
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# # ==================================================================

# Function to load and resize images from a folder
def load_and_resize_images(folder):
    X = []
    y = []
    class_folders = sorted(os.listdir(folder))
    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.npy'):
                    img_path = os.path.join(class_path, filename)
                    feature = np.load(img_path)  
                    X.append(feature)
                    y.append(class_folder)
    return np.array(X), np.array(y)


# # ==================================================================
# Function to normalize the data using Gaussian normalization
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

# ====================== MAIN =====================================

# Load and resize train set
train_images, train_labels = load_and_resize_images(train_folder)
# Load and resize test set
test_images, test_labels = load_and_resize_images(test_folder)

# Reshape feature data to 1D vectors
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
    
# Normalize the feature data using Gaussian normalization
train_images_normalized, test_images_normalized = normalize_data(train_images, test_images)
# Split training data into training, validation, and testing sets
test_images_normalized, val_images_normalized, test_labels, val_labels = train_test_split(test_images_normalized, test_labels, test_size=0.5, random_state=42)

input_dim = train_images_normalized.shape[1]  # Get the number of features from your data
num_classes = num_categories  # Number of classification categories

# Save the model configurations to CSV
save_model_configurations_to_csv(input_dim, num_classes, result_folder)


# Convert labels to categorical
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
val_labels_encoded = label_encoder.transform(val_labels)

num_categories = len(label_encoder.classes_)  # Get the number of unique classes

train_labels_categorical = to_categorical(train_labels_encoded, num_classes=num_categories)
test_labels_categorical = to_categorical(test_labels_encoded, num_classes=num_categories)
val_labels_categorical = to_categorical(val_labels_encoded, num_classes=num_categories)

class_labels = label_encoder.classes_

model_0_metrics = []
model_1_metrics = []
model_2_metrics = []
stacked_model_metrics = []

# Ensure this is done after label encoding and before your loop for ROC calculation
test_labels_multilabel = to_categorical(test_labels_encoded, num_classes=len(label_encoder.classes_))

# Assuming train_images_normalized and train_labels have been appropriately prepared
input_dim = train_images_normalized.shape[1]  # Number of features
num_classes = np.unique(train_labels).size    # Number of unique classes

# Create each model using the appropriate function
model_0 = create_keras_model_1(input_dim, num_classes)
model_1 = create_keras_model_2(input_dim, num_classes)
model_2 = create_keras_model_3(input_dim, num_classes)


for batch_size in batch_sizes:
    # Create and compile your model
    batch_size_result_out = os.path.join(result_folder, 'batch_size_' + str(batch_size))
    os.makedirs(batch_size_result_out, exist_ok=True)
    
    for epoch in epoch_values:
        result_out = os.path.join(batch_size_result_out, 'epoch_'+ str(epoch))
        os.makedirs(result_out, exist_ok=True)
        
        # model_0 = create_keras_model_1(train_images_normalized, train_labels)
        # model_1 = create_keras_model_2(train_images_normalized, train_labels)
        # model_2 = create_keras_model_3(train_images_normalized, train_labels)

        # ------------------
        start_time = time.time()
        history_0 = model_0.fit(train_images_normalized, train_labels_categorical, epochs=epoch, batch_size=32, 
                                validation_data=(val_images_normalized, val_labels_categorical), verbose=2)
        training_time_0 = time.time() - start_time
        model_0.save(os.path.join(result_out, 'model_0.h5'))
        # ------------------
        start_time = time.time()
        history_1 = model_1.fit(train_images_normalized, train_labels_categorical, epochs=epoch, batch_size=32, 
                                validation_data=(val_images_normalized, val_labels_categorical), verbose=2)
        training_time_1 = time.time() - start_time
        model_1.save(os.path.join(result_out, 'model_1.h5'))
        
        # ------------------
        start_time = time.time()
        history_2 = model_2.fit(train_images_normalized, train_labels_categorical, epochs=epoch, batch_size=32, 
                                validation_data=(val_images_normalized, val_labels_categorical), verbose=2)
        training_time_2 = time.time() - start_time
        model_2.save(os.path.join(result_out, 'model_2.h5'))

        
        #==================================================
        # Generate cconfusion_matrix for model_1
        confusion_matrix_report(model_0, test_images_normalized, test_labels_encoded, result_out, 'model_0')
        # Generate cconfusion_matrix for model_2
        confusion_matrix_report(model_1, test_images_normalized, test_labels_encoded, result_out, 'model_1')
        # Generate cconfusion_matrix for model_3
        confusion_matrix_report(model_2, test_images_normalized, test_labels_encoded, result_out, 'model_2')
        # Plot the training history for each model
        #==================================================
        accuracy_loss_plot(result_out, epoch, history_0, 'model_0')
        accuracy_loss_plot(result_out, epoch, history_1, 'model_1')
        accuracy_loss_plot(result_out, epoch, history_2, 'model_2')
        #==================================================
        
        #================== Build Stack ================================
        n_members = 3
        members = load_all_models(n_members, result_out)
        print('Loaded %d models' % len(members))
        
        # Call the function with the filename you want
        save_to_csv(val_images_normalized, val_labels_encoded, result_out)
        
        start_time = time.time()
        stacked_model = fit_stacked_model(members, val_images_normalized, val_labels_categorical)
        stacked_training_time = time.time() - start_time
        
        model_0_metrics.append(collect_metrics(model_0, test_images_normalized, test_labels_encoded, result_out, 'model_0', epoch, batch_size, training_time_0))
        model_1_metrics.append(collect_metrics(model_1, test_images_normalized, test_labels_encoded, result_out, 'model_1', epoch, batch_size, training_time_1))
        model_2_metrics.append(collect_metrics(model_2, test_images_normalized, test_labels_encoded, result_out, 'model_2', epoch, batch_size, training_time_2))

        # Prediction and metrics collection:
        # Generate predictions using the stacked model
        y_pred_stack = stacked_prediction(members, stacked_model, test_images_normalized)
        y_pred_stack_multiclass = np.argmax(y_pred_stack, axis=1)


        
        # You might also want to calculate metrics here if needed
        stacked_metrics = collect_stacked_model_metrics(
            members=members,
            stacked_model=stacked_model,
            X_test=test_images_normalized,
            y_test=test_labels_encoded,
            result_out=result_out,
            model_name='Stacked_Model',
            batch_size=batch_size,
            epoch=epoch,
            training_time=stacked_training_time
        )
        
        # Add the collected metrics to a list or directly to a DataFrame
        stacked_model_metrics.append(stacked_metrics)

        # Generate the confusion matrix for the stacked model
        confusion_matrix_stacked = confusion_matrix(test_labels_encoded, y_pred_stack_multiclass)
        
        # Plot the confusion matrix with class labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_stacked, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
        plt.savefig(os.path.join(result_out, 'stack_confusion_matrix.png'))
        plt.close()
        
        plot_stack_roc_curves(test_labels_categorical, y_pred_stack, result_out, 'Stacked_Model', class_labels)
        #==================================================  


# Convert lists to DataFrame
df_model_0 = pd.DataFrame(model_0_metrics)
df_model_1 = pd.DataFrame(model_1_metrics)
df_model_2 = pd.DataFrame(model_2_metrics)
df_stacked = pd.DataFrame(stacked_model_metrics)

# Define file paths
path_model_0 = os.path.join(result_folder, f'{model_name}_model_0_metrics.csv')
path_model_1 = os.path.join(result_folder, f'{model_name}_model_1_metrics.csv')
path_model_2 = os.path.join(result_folder, f'{model_name}_model_2_metrics.csv')
path_stacked = os.path.join(result_folder, f'{model_name}_stacked_model_metrics.csv')

# Save to CSV
df_model_0.to_csv(path_model_0, index=False)
df_model_1.to_csv(path_model_1, index=False)
df_model_2.to_csv(path_model_2, index=False)
df_stacked.to_csv(path_stacked, index=False)


