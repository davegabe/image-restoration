from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': 'images'})
filters = dict(
    # color='orange',
    # license='commercial,modify',
    # date=((2017, 1, 1), (2017, 11, 30))
    )
google_crawler.crawl(keyword='dog imagesize:900x900', filters=filters, max_num=5, file_idx_offset=0)
