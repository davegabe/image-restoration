import os

from icrawler.builtin import GoogleImageCrawler
from PIL import Image


def download(training_path: str, evaluation_path: str, width: int, height: int, keyword: str, quantity: int):
    """
    Downloads training data from the internet and saves it to the specified destination, separated into training and evaluation data.

    Args:
        training_path: The path to save the training data.
        evaluation_path: The path to save the evaluation data.
    """

    google_crawler = GoogleImageCrawler(
        storage={'root_dir': training_path}
    )

    filters = {'type': "photo"}
    google_crawler.crawl(keyword=keyword, filters=filters, max_num=quantity,
                         min_size=(width, height), file_idx_offset='auto')

    adjustAndCrop(training_path, width, height)
    splitData(training_path, evaluation_path)


def adjustAndCrop(path: str, width: int, height: int):
    """
    Adjusts and crops images to the specified size.

    Args:
        path: The path to the images.
        width: The width of desired images. If width=0, it's the smallest possible size.
        height: The height of desired images. If height=0, it's the smallest possible size.
    """
    files = os.listdir(path)

    # if width or height is 0, it's the smallest possible size
    if width == 0 or height == 0:
        tmp_width = 0
        tmp_height = 0
        for file in files:
            if not (file.startswith('.')):  # ignore system files
                image = Image.open(os.path.join(path, file))
                if tmp_width == 0:
                    tmp_width = image.width
                else:
                    if image.width < tmp_width:
                        tmp_width = image.width
                if tmp_height == 0:
                    tmp_height = image.height
                else:
                    if image.height < tmp_height:
                        tmp_height = image.height
                image.close()
        if width == 0:
            width = tmp_width
        if height == 0:
            height = tmp_height

    # random crop the images to the specified size using the PIL library
    for file in files:
        img = Image.open(os.path.join(path, file))

        # resize the image as the minimum size is (width, height)
        if img.width > width and img.height > height:
            if img.size[0] > img.size[1]:
                new_img = img.resize(
                    (int(img.size[0] * height / img.size[1]), height), Image.ANTIALIAS)
            else:
                new_img = img.resize(
                    (width, int(img.size[1] * width / img.size[0])), Image.ANTIALIAS)

            new_img = new_img.crop((0, 0, width, height)) # crop the image to the specified size
            img.close()
            new_img.save(os.path.join(path, file)) # overwrite the old image
        else:
            # if the image is smaller than the specified size, remove it
            img.close()
            os.remove(os.path.join(path, file))


def splitData(training_path: str, evaluation_path: str):
    """
        Splits the downloaded data into training and evaluation data.
        Args:
            training_path: The path to the training data.
            evaluation_path: The path to the evaluation data.
    """
    files = os.listdir(training_path)

    for i in range(int(len(files)*0.2)):  # move 20% of images to evaluation path
        old_file_path = os.path.join(training_path, files[i])
        new_file_path = os.path.join(evaluation_path, files[i])
        if not (files[i].startswith('.')):  # ignore system files
            os.rename(old_file_path, new_file_path)
