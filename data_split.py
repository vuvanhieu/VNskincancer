import os
import random
import shutil

directory_work = os.getcwd()
path = os.path.join(directory_work, 'OutputAugentation')

subfolder_names = ['BCC', 'MM', 'SCC', 'no skin cancer']

train_dir = os.path.join(directory_work, 'train')
test_dir = os.path.join(directory_work, 'test')

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the percentage split
test_percentage = 0.2  # 20% for testing, 80% for training

for label in subfolder_names:
    label_dir = os.path.join(path, label)
    label_train_dir = os.path.join(train_dir, label)
    label_test_dir = os.path.join(test_dir, label)

    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    # List all the files in the label directory
    files = os.listdir(label_dir)
    
    # Calculate the number of files to copy to the test directory
    num_test_files = int(len(files) * test_percentage)
    
    # Randomly select files for testing
    test_files = random.sample(files, num_test_files)
    
    for file in files:
        source_path = os.path.join(label_dir, file)
        if file in test_files:
            destination_path = os.path.join(label_test_dir, file)
        else:
            destination_path = os.path.join(label_train_dir, file)
        
        # Copy the file to the appropriate directory
        shutil.copy(source_path, destination_path)

print("Data split completed.")
