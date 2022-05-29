from icrawler.builtin import GoogleImageCrawler
import splitfolders

def download(training_path, evaluation_path, width, height, keyword, number):
    """
    Downloads training data from the internet and saves it to the specified destination, separated into training and evaluation data.
    
    Args:
        training_path: The path to save the training data.
        evaluation_path: The path to save the evaluation data.
    """
    
    keyword = keyword+" imagesize:" + width + "x" + height

    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': training_path})

    google_crawler.crawl(keyword=keyword, max_num=number, file_idx_offset=0)
    
    print(training_path, evaluation_path)
    pass
