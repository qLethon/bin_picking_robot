from PIL import Image, ImageEnhance
import os
import argparse

def change_brightness(source_dir, save_dir, brightness):
    os.makedirs(save_dir, exist_ok=True)
    image_pathes = [f for f in os.scandir(source_dir) if f.is_file()]
    for image_path in image_pathes:
        save_path = os.path.join(save_dir, image_path.name.rstrip('.jpg') + "_" + str(brightness) + ".jpg")
        ImageEnhance.Brightness(Image.open(image_path.path)).enhance(brightness).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagedir', type=str, required=True)
    parser.add_argument('-s', '--savedir', type=str, required=True)
    args = parser.parse_args()

    
    for brightness in (0.5, 0.75, 1.25, 1.5):
        change_brightness(args.imagedir, os.path.join(args.savedir, str(brightness)), brightness)