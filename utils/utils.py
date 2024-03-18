from PIL import Image

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(
        image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    # find top M * N * p indexes
    searchidx = (-flatdark).argsort()[:int(M * N * p)]
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    if img.width > 750:
        new_width = 750
        new_height = int(new_width * img.height / img.width)
        img = img.resize((new_width, new_height))

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def save_images_pytorch(results, transform, size, name, folder='./images/prediction/'):
    index = 1
    for _, _, pr_image_batch in results:
        for pr_image in pr_image_batch:
            out_name = folder + '{}_{}_{}.png'.format(size, index, name)
            transform(pr_image).save(out_name)
            index += 1


def save_images_tensorflow(model, sequence, size, name, folder='./images/prediction/'):
    index = 1
    for haze, _ in sequence:
        pred = model.predict(haze)
        for pr in pred:
            image = (pr - np.min(pr)) / (np.max(pr) - np.min(pr))
            image = Image.fromarray((image * 255).astype(np.uint8))
            out_name = folder + '{}_{}_{}.png'.format(size, index, name)
            image.save(out_name)
            index += 1


def get_information(historial, collection):
    _, mssim, mpsnr = np.mean(historial, axis=0)
    _, issim, ipsnr = np.argmax(historial, axis=0)

    print("Mean SSIM:", mssim)
    print("Mean PSNR:", mpsnr)
    print("Best SSIM:", collection.images[issim][0])
    print("Best PSNR:", collection.images[ipsnr][0])
