import os
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from PIL import Image

def create_augmentation_mapping(csv_file_path):
    df = pd.read_csv(csv_file_path)
    mapping = df.set_index('Category')['Total_Folds_Exact'].fillna(1).astype(int).to_dict()
    return mapping

def define_augmentation_sequence():
    return iaa.Sequential([
        iaa.Affine(rotate=(-25, 25), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Flipud(0.5), iaa.Fliplr(0.5),
        iaa.Multiply((0.8, 1.2)),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        iaa.ContrastNormalization((0.5, 2.0)),
        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    ])

def process_images(directory_path, output_dir, augmentation_count):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    seq = define_augmentation_sequence()
    for image_file in os.listdir(directory_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, image_file)
            image = load_img(image_path)
            image_array = img_to_array(image)
            image_array = np.clip(image_array, 0, 255)  # Ensure image is in the correct range
            image_array = image_array.astype(np.uint8)  # Convert image to uint8

            for i in range(augmentation_count):
                image_aug = seq(image=image_array.copy())
                aug_image_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_aug_{i}.jpg')
                save_img(aug_image_path, image_aug)

def main(input_dir, output_dir, csv_file_path):
    augmentation_mapping = create_augmentation_mapping(csv_file_path)
    categories = ['BCC', 'MM', 'SCC', 'no skin cancer']
    for category in categories:
        print(f'Processing category: {category}')
        category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        num_augmentations = augmentation_mapping.get(category, 1)
        process_images(category_dir, output_category_dir, num_augmentations)

input_directory = 'inputImages'
output_directory = 'OutputAugentation'
csv_path = 'Augmentation_Applied.csv'

main(input_directory, output_directory, csv_path)
