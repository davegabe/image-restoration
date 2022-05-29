from icrawler.builtin import GoogleImageCrawler
import os

def download(training_path, evaluation_path, width, height, keyword, number):
    """
    Downloads training data from the internet and saves it to the specified destination, separated into training and evaluation data.

    Args:
        training_path: The path to save the training data.
        evaluation_path: The path to save the evaluation data.
    """

    keyword = keyword+" imagesize:" + str(width) + "x" + str(height)

    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': evaluation_path})

    google_crawler.crawl(keyword=keyword, max_num=number, file_idx_offset=0)

    files = os.listdir(evaluation_path)

    for i in range(int(len(files)*0.8)):  # move 80% of images to training path
        old_file_path = os.path.join(evaluation_path, files[i])
        new_file_path = os.path.join(training_path, files[i])
        if not (files[i].startswith('.')): # ignore system files
            os.rename(old_file_path, new_file_path)