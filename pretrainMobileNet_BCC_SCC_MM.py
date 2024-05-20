import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

directory_work = os.getcwd()
experiment_folder = os.path.join(directory_work, 'Multimedia_Tools_and_Applications_SkinLesion_BCC_SCC_MM_(before_Review)')
os.makedirs(experiment_folder, exist_ok=True)

model_name = 'pretrainMobileNet_BCC_SCC_MM'
result_folder = os.path.join(experiment_folder, model_name)
os.makedirs(result_folder, exist_ok=True)

# Set the paths to the train and test folders
train_folder = "/data2/cmdir/home/anhnv/vvhieu/Hien/Hien_Data_Feature/train"
test_folder = "/data2/cmdir/home/anhnv/vvhieu/Hien/Hien_Data_Feature/test"

# Define the categories
categories = ['BCC', 'MM', 'SCC']
num_categories = len(categories)

# Function to load image files from a folder
def load_and_resize_images(folder, target_size=(224, 224), image_extensions=('.jpg', '.jpeg', '.png')):
    images = []
    labels = []
    names = []
    for category in categories:
        category_folder = os.path.join(folder, category)
        for filename in os.listdir(category_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(category_folder, filename)
                label = category
                img = load_img(img_path, target_size=target_size)  # Load and resize the image
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
                names.append(filename)
    return np.array(images), np.array(labels), names

# Function to get the shape of each category in a folder
def get_category_shapes(folder, categories):
    category_shapes = []
    for category in categories:
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            image_files = [f for f in os.listdir(category_path) if any(f.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png'))]
            num_images = len(image_files)
            category_shapes.append({'Category': category, 'Number of Images': num_images})
    return category_shapes

# Load images for train and test datasets
train_images, train_labels, _ = load_and_resize_images(train_folder)
test_images, test_labels, test_file_paths = load_and_resize_images(test_folder)

# Debugging: Check if images are loaded correctly
print(f'Loaded {len(train_images)} training images.')
print(f'Loaded {len(test_images)} testing images.')

# Check if images were loaded successfully
if len(train_images) == 0 or len(test_images) == 0:
    raise ValueError("No images were loaded. Check the file paths and formats of the images in the folders.")

# Get the shape of each category in the train and test folders
train_category_shapes = get_category_shapes(train_folder, categories)
test_category_shapes = get_category_shapes(test_folder, categories)

# Combine train and test category shapes
train_category_shapes = [{'Category': shape['Category'], 'Number of Images (Train)': shape['Number of Images']} for shape in train_category_shapes]
test_category_shapes = [{'Category': shape['Category'], 'Number of Images (Test)': shape['Number of Images']} for shape in test_category_shapes]

# Merge the two lists on the Category key
category_shapes_df = pd.DataFrame(train_category_shapes).merge(pd.DataFrame(test_category_shapes), on='Category', how='outer')
category_shapes_csv_path = os.path.join(result_folder, 'category_shapes.csv')
category_shapes_df.to_csv(category_shapes_csv_path, index=False)

print('Category shapes saved to:', category_shapes_csv_path)

# Shuffle the train data
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

# Split test data to use 50% for validation during training
test_images, val_images, test_labels, val_labels, test_file_paths, val_file_paths = train_test_split(
    test_images, test_labels, test_file_paths, test_size=0.5, random_state=42)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Debugging: Check label distribution
print("Train label distribution:", np.bincount(train_labels))
print("Validation label distribution:", np.bincount(val_labels))
print("Test label distribution:", np.bincount(test_labels))

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels, num_categories)
val_labels = to_categorical(val_labels, num_categories)
test_labels = to_categorical(test_labels, num_categories)

# Debugging: Check shape of one-hot encoded labels
print("Shape of train labels:", train_labels.shape)
print("Shape of validation labels:", val_labels.shape)
print("Shape of test labels:", test_labels.shape)

# Build the model using MobileNet
base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base.layers:
    layer.trainable = False
    print(layer.name)

x = GlobalMaxPooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predict = Dense(num_categories, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predict)

optim = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_aug = ImageDataGenerator(rotation_range=60,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               fill_mode='nearest')

val_aug = ImageDataGenerator()

# Set all layers to be trainable
for layer in base.layers:
    layer.trainable = True

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5,
                                            min_lr=0.000001, cooldown=2)

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

epoch_values = [3, 10, 20, 40, 60, 80, 100]
metrics_list = []

for epoch in epoch_values:
    result_out = os.path.join(result_folder, 'epoch_' + str(epoch))
    os.makedirs(result_out, exist_ok=True)
    
    start_time = datetime.now()  # Record start time
    history_0 = model.fit(train_images, train_labels, epochs=epoch, batch_size=64, validation_data=(val_images, val_labels), verbose=2)
    end_time = datetime.now()  # Record end time
    time_taken_0 = end_time - start_time

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    # Get predictions on test data
    predictions = model.predict(test_images, verbose=0)
    y_pred_classes = np.argmax(predictions, axis=1)

    # Convert numerical labels back to string labels
    y_true_labels = label_encoder.inverse_transform(np.argmax(test_labels, axis=1))
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
    
    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='macro')
    
    # Calculate sensitivity and specificity for each class
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    sensitivity_0 = np.mean(np.diag(cm) / np.sum(cm, axis=1))  # Mean sensitivity
    specificity_0 = np.mean([cm[i, i] / (np.sum(cm[:, i]) or 1) for i in range(len(cm))])  # Mean specificity
    
    # Calculate AUC score
    auc_score = roc_auc_score(test_labels, predictions, average='macro')
    
    # Generate classification report
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    
    # Save classification report to a CSV file
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(result_out, model_name + '_classification_report.csv')
    report_df.to_csv(report_csv_path)
    
    # Print a message indicating the file location
    print('Classification report saved to:', report_csv_path)

    # Plot accuracy
    plt.plot(history_0.history['accuracy'], linestyle='--')
    plt.plot(history_0.history['val_accuracy'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, model_name + '_accuracy_plot.png'))
    plt.show()
    plt.close()

    # Plot loss
    plt.plot(history_0.history['loss'], linestyle='--')
    plt.plot(history_0.history['val_loss'], linestyle=':')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, model_name + '_loss_plot.png'))
    plt.show()
    plt.close()

    # Compute the ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_categories):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
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
    plt.savefig(os.path.join(result_out, model_name + '_roc_curve.png'))
    plt.show()
    plt.close()

    # Compute the confusion matrix
    confusion_mat = confusion_matrix(y_true_labels, y_pred_labels, labels=categories)

    # Plot confusion matrix with labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + '_confusion_matrix.png'))
    plt.show()
    plt.close()
    
    # Update metrics
    metrics = {
        'Model Name': model_name,
        'Batch Size': 64,
        'Epoch': epoch,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Sensitivity': sensitivity_0,
        'Specificity': specificity_0,
        'AUC Score': auc_score,
        'Loss': loss,
        'Epoch Training Time': time_taken_0
    }
    metrics_list.append(metrics)

# Save the metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_csv_path = os.path.join(result_folder, f'{model_name}_metrics_summary.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

print('Metrics summary saved to:', metrics_csv_path)

# Predict all images in the test_folder using the trained model
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels_str = label_encoder.inverse_transform(predicted_labels)

# Get the true labels for the test set
true_labels_str = label_encoder.inverse_transform(np.argmax(test_labels, axis=1))

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'File Path': test_file_paths,
    'True Label': true_labels_str,
    'Predicted Label': predicted_labels_str
})
predictions_csv_path = os.path.join(result_folder, f'{model_name}_predictions_on_test_set.csv')
predictions_df.to_csv(predictions_csv_path, index=False)

print('Predictions on test set saved to:', predictions_csv_path)
