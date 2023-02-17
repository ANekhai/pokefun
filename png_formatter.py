from glob import glob
from PIL import Image

def convert_png_to_rgb(folder):
    for filename in glob(folder + "/*.png"):
        print(f"Processing: {filename}")
        image = Image.open(filename)
        if image.mode == "P":
            image = image.convert("RGBA")
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, (0, 0), image)
            bg.save(filename)
        elif image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, (0, 0), image)
            bg.save(filename)

if __name__ == "__main__":

    folders = ["sprites/female_sprites", "sprites/front_sprites", "sprites/shiny_female_sprites", "sprites/shiny_sprites"]

    for folder in folders:
        convert_png_to_rgb(folder)
        # rgba_images = set([filename for filename in glob(folder + "/*.png") if Image.open(filename).mode == "RGBA"])
        # print(rgba_images)
