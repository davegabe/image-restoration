from bing_image_downloader import downloader

#https://pypi.org/project/bing-image-downloader/

search_queries = [
    "yoyoyo"
]

def download_images(queries):
    for query in queries:
        downloader.download(query, limit=5, output_dir = "dataset", adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
        print()

download_images(search_queries)