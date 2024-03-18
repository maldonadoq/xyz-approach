import os

from PIL import Image
from .utils import pil_to_np, crop_image


class HazeCollection:
    def __init__(self, haze_dir, image_dir=None, transform=None):
        self.haze_dir = haze_dir
        self.image_dir = image_dir
        self.transform = transform

        self.images = []
        self.pair_images()

    def pair_images(self):
        if self.image_dir is None:
            for haze in os.listdir(self.haze_dir):
                self.images.append((haze, None))
        else:
            matching_dict = dict()
            for haze in os.listdir(self.haze_dir):
                underlines = haze.split('_')
                image = underlines[0] + '.' + haze.split('.')[-1] if len(underlines) > 1 else haze

                if image in matching_dict:
                    matching_dict[image].append(haze)
                else:
                    matching_dict[image] = [haze]
            for image in matching_dict:
                for haze in matching_dict[image]:
                    self.images.append((haze, image))

    def __len__(self):
        return len(self.images)

    def get_image(self, path):
        image = Image.open(path).convert("RGB")
        image = image if self.transform is None else self.transform(image)
        image = crop_image(image, d=32)
        image = pil_to_np(image)

        return image

    def __getitem__(self, index):
        haze_path = os.path.join(self.haze_dir, self.images[index][0])
        haze = self.get_image(haze_path)

        image = None
        if self.image_dir:
            image_path = os.path.join(self.image_dir, self.images[index][1])
            image = self.get_image(image_path)

        return haze, image, self.images[index]
