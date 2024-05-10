# VNskincancer


## OutputAugentation.py:
1. Read the CSV File: Determines how many augmentations each category of images should receive.
2. Define Augmentation Techniques: Establishes a comprehensive set of transformations that can be applied to the images.
3. Apply Augmentations: For each category and for each image, apply the defined augmentations and save the results in a structured way.

## data_split.py
Data Splitting Process
1. Directory Preparation: For each category (e.g., 'BCC', 'MM', etc.):
+ Paths for each category's train and test directories are generated and created.
+ Retrieves all filenames in the category's original directory.
+ Calculates the number of files to allocate to the test set based on the specified test percentage.
2. File Distribution:
+ Randomly selects a specified number of files to include in the test set using random.sample().
+ Iterates over all files in the current category's directory:
+ Determines whether each file should go into the train or test directory based on whether the file is in the list of selected test files.
+ Copies the file to the appropriate directory using shutil.copy(), which copies the file from the source path to the destination path.
