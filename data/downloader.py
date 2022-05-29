from icrawler.builtin import GoogleImageCrawler
import splitfolders

def download(training_path, evaluation_path, width, height, keyword, number):
    """
    Downloads training data from the internet and saves it to the specified destination, separated into training and evaluation data.
    
    Args:
        training_path: The path to save the training data.
        evaluation_path: The path to save the evaluation data.
    """
    
    keyword1 = keyword+" imagesize:" + width + "x" + height
    keyword2 = keyword+ "s" + " imagesize:" + width + "x" + height      #plural just to have different images

    google_crawler_train = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': training_path})
    
    google_crawler_eval = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': evaluation_path})

    google_crawler_train.crawl(keyword=keyword1, max_num=number*0.8, file_idx_offset=0)

    google_crawler_eval.crawl(keyword=keyword2, max_num=number*0.2, file_idx_offset=0)
    
    print(training_path, evaluation_path)
    pass
