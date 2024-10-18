import argparse
import os
import random
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Path to the main directory containing character folders
base_dir = os.path.join(".", "Custom_Dot_Matrix_Dataset", "Dot_Matrix_Test_1")

# Path to the background image
background_image_path = os.path.join(".", "background")


def registry_chars(base_dir):
    char_registry = {
        "0": (os.path.join(base_dir, "0"), 0),
        "1": (os.path.join(base_dir, "1"), 1),
        "2": (os.path.join(base_dir, "2"), 2),
        "3": (os.path.join(base_dir, "3"), 3),
        "4": (os.path.join(base_dir, "4"), 4),
        "5": (os.path.join(base_dir, "5"), 5),
        "6": (os.path.join(base_dir, "6"), 6),
        "7": (os.path.join(base_dir, "7"), 7),
        "8": (os.path.join(base_dir, "8"), 8),
        "9": (os.path.join(base_dir, "9"), 9),
        "A": (os.path.join(base_dir, "A"), 10),
        "B": (os.path.join(base_dir, "B"), 11),
        "C": (os.path.join(base_dir, "C"), 12),
        "D": (os.path.join(base_dir, "D"), 13),
        "E": (os.path.join(base_dir, "E"), 14),
        "F": (os.path.join(base_dir, "F"), 15),
        "G": (os.path.join(base_dir, "G"), 16),
        "H": (os.path.join(base_dir, "H"), 17),
        "I": (os.path.join(base_dir, "I"), 18),
        "J": (os.path.join(base_dir, "J"), 19),
        "K": (os.path.join(base_dir, "K"), 20),
        "L": (os.path.join(base_dir, "L"), 21),
        "M": (os.path.join(base_dir, "M"), 22),
        "N": (os.path.join(base_dir, "N"), 23),
        "O": (os.path.join(base_dir, "O"), 24),
        "P": (os.path.join(base_dir, "P"), 25),
        "Q": (os.path.join(base_dir, "Q"), 26),
        "R": (os.path.join(base_dir, "R"), 27),
        "S": (os.path.join(base_dir, "S"), 28),
        "T": (os.path.join(base_dir, "T"), 29),
        "U": (os.path.join(base_dir, "U"), 30),
        "V": (os.path.join(base_dir, "V"), 31),
        "W": (os.path.join(base_dir, "W"), 32),
        "X": (os.path.join(base_dir, "X"), 33),
        "Y": (os.path.join(base_dir, "Y"), 34),
        "Z": (os.path.join(base_dir, "Z"), 35),
        ":": (os.path.join(base_dir, "Colon"), 36),
        "/": (os.path.join(base_dir, "Slash"), 37),
    }
    return char_registry


def put_img_on_background(char_img: Image.Image, bg_img, position):
    bg_img.paste(
        char_img, position, char_img.convert("RGBA")
    )  # Use RGBA for transparency support


def convert_img_to_8bit_non_background(img_path):
    # Read the image using OpenCV
    tiff_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Normalize the image to 8-bit
    image_8bit = (tiff_image / tiff_image.max() * 255).astype(np.uint8)

    # Convert the image to RGBA format
    image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_BGRA2RGBA)

    # Remove the background (assuming the background is white)
    # Set all white pixels to transparent
    image_8bit[np.all(image_8bit[:, :, :3] == 255, axis=-1)] = [128, 128, 128, 0]
    alpha = image_8bit[:, :, 3]
    xmin = np.min(np.where(alpha > 0)[1]) - 2
    xmax = np.max(np.where(alpha > 0)[1]) + 4
    ymin = np.min(np.where(alpha > 0)[0]) - 2
    ymax = np.max(np.where(alpha > 0)[0]) + 4
    image_8bit = image_8bit[ymin:ymax, xmin:xmax]
    return image_8bit


# Define where to place the character images on the background
# (You can adjust this or make it dynamic)
def get_position_for_character(index, img_size, bg_size, offset_x, offset_y):
    # Calculate position to center the image (or modify as needed)
    x = offset_x
    y = offset_y
    return (x, y)


def analyze_sentence(sentence, char_registry):
    chars = list(sentence)
    char_list = list()
    count = -len(chars) // 2
    for char in chars:
        if char in char_registry:
            char_path = random.choice(os.listdir(char_registry[char][0]))
            char_list.append(
                (count, char, os.path.join(char_registry[char][0], char_path))
            )
            count += 1
        else:
            print(f"Character {char} not found in the registry")
    return char_list


def generate_annotations(char_registry, char, char_img, bg_img, position):
    x1, y1 = position[0] / bg_img.size[0], position[1] / bg_img.size[1]
    x2, y2 = x1 + char_img.size[0] / bg_img.size[0], y1
    x3, y3 = x2, y2 + char_img.size[1] / bg_img.size[1]
    x4, y4 = x1, y3
    class_id = char_registry[char][1]
    return f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"


def generate_expiration_date():
    start_date = datetime.now()
    end_date = start_date + timedelta(days=random.randint(30, 365))
    sentence = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    return sentence + ":" + end_date.strftime("%m/%d/%Y")


def arg_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(".", "Custom_Dot_Matrix_Dataset", "Dot_Matrix_Test_1"),
        help="Path to the main directory containing character folders",
    )
    parser.add_argument(
        "--background_image_path",
        type=str,
        default=os.path.join(".", "background"),
        help="Path to the background image",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    return parser.parse_args()


def main():
    args = arg_paser()
    char_registry = registry_chars(args.base_dir)
    background_image_path = args.background_image_path
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    annotations_dir = os.path.join(args.output_dir, "annotations")
    if os.path.exists(images_dir) is False:
        os.makedirs(images_dir, exist_ok=True)
    if os.path.exists(annotations_dir) is False:
        os.makedirs(os.path.join(annotations_dir, "annotations"), exist_ok=True)
    for k in range(1, 101):
        sentence = generate_expiration_date()
        char_list = analyze_sentence(sentence, char_registry)
        bg_img = Image.open(
            os.path.join(
                background_image_path, random.choice(os.listdir(background_image_path))
            )
        )
        offset_y = bg_img.size[1] // 3
        offset_x = bg_img.size[0] // 4

        sentence_ = generate_expiration_date()
        char_list_ = analyze_sentence(sentence_, char_registry)
        offset_y_ = offset_y + 77
        offset_x_ = bg_img.size[0] // 4

        # with open(f"./annotations/random_result_{k}.txt", "w") as f:
        with open(os.path.join(annotations_dir, f"random_result_{k}.txt"), "w") as f:
            for i, char, path in char_list:
                char_img = convert_img_to_8bit_non_background(path)

                # char_img = cut_width_of_char_image(char_img, cut_ratio=0.7)
                char_img[:, :, 3] = char_img[:, :, 3] * random.uniform(0.8, 0.9)
                char_img = Image.fromarray(char_img)
                char_img = char_img.filter(ImageFilter.GaussianBlur(radius=1))
                position = get_position_for_character(
                    i, char_img.size, bg_img.size, offset_x, offset_y
                )
                offset_x = offset_x + char_img.size[0]
                distance = random.randint(10, 15)
                offset_x = offset_x + distance

                put_img_on_background(char_img, bg_img, position)
                f.write(
                    generate_annotations(
                        char_registry, char, char_img, bg_img, position
                    )
                )
                f.write("\n")

            for i, char, path in char_list_:
                char_img = convert_img_to_8bit_non_background(path)

                # char_img = cut_width_of_char_image(char_img, cut_ratio=0.7)
                char_img[:, :, 3] = char_img[:, :, 3] * random.uniform(0.8, 0.9)
                char_img = Image.fromarray(char_img)
                char_img = char_img.filter(ImageFilter.GaussianBlur(radius=1))
                position = get_position_for_character(
                    i, char_img.size, bg_img.size, offset_x_, offset_y_
                )
                offset_x_ = offset_x_ + char_img.size[0]
                distance = random.randint(10, 15)
                offset_x_ = offset_x_ + distance

                put_img_on_background(char_img, bg_img, position)
                f.write(
                    generate_annotations(
                        char_registry, char, char_img, bg_img, position
                    )
                )
                f.write("\n")
        # bg_img.save(f"./images/random_result_{k}.png")
        bg_img.save(os.path.join(images_dir, f"random_result_{k}.png"))
    print("Done")


if __name__ == "__main__":
    main()
