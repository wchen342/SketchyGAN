import logging
import json
import os
from time import time
from six.moves.urllib.parse import urlparse
from datetime import date, timedelta

from icrawler.downloader import ImageDownloader
from icrawler.builtin import FlickrImageCrawler
from icrawler.builtin.flickr import FlickrParser


# You need a Flickr API key to make this script work. Also note that Flickr has a rate restriction
# so you cannot crawl too fast.


# Override default icrawler classes
class MyFlickrParser(FlickrParser):

    def parse(self, response, apikey, size_preference=None):
        content = json.loads(response.content.decode())
        if content['stat'] != 'ok':
            raise ValueError("Status: %s" % content['stat'])
        photos = content['photos']['photo']
        print('Num photos: %d' % len(photos))
        for photo in photos:
            photo_id = photo['id']

            if 'url_z' in photo.keys():
                url = photo['url_z']
            elif 'url_n' in photo.keys():
                url = photo['url_n']
            else:
                # print('Empty URL!')
                continue

            yield dict(file_url=url, meta=photo)


class MyImageDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        """Set the path where the image will be saved.

        The default strategy is to use an increasing 6-digit number as
        the filename. You can override this method if you want to set custom
        naming rules. The file extension is kept if it can be obtained from
        the url, otherwise ``default_ext`` is used as extension.

        Args:(i + 1)
            task (dict): The task dict got from ``task_queue``.

        Output:
            Filename with extension.
        """
        url_path = urlparse(task['file_url'])[2]
        extension = url_path.split('.')[-1] if '.' in url_path else default_ext
        filename = url_path.split('.')[0].split('/')[-1].split('_')[0]
        # file_idx = self.fetched_num + self.file_idx_offset
        return '{}.{}'.format(filename, extension)


TODAY = date(2018, 4, 21)
delta = timedelta(days=5 * 365/12)    # go back 5 years
output_path = '../flickr_output'


# Main method
def crawl(crawl_list, work_list):
    for i in range(len(crawl_list)):
        class_name = crawl_list[i]
        print("Now fetching class: %s" % class_name)
        output_dir = os.path.join(output_path, class_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        flickr_crawler = FlickrImageCrawler('',    # put your Flickr API key here
                                            feeder_threads=2, parser_threads=10, downloader_threads=5,
                                            parser_cls=MyFlickrParser,
                                            downloader_cls=MyImageDownloader,
                                            storage={'root_dir': output_dir},
                                            log_level=logging.ERROR)

        # Time counter
        prev_time = float("-inf")
        curr_time = float("-inf")
        for i in range(28):
            curr_time = time()
            elapsed = curr_time - prev_time
            print(
                "Now at iteration %d. Elapsed time: %.5fs." % (i, elapsed))
            prev_time = curr_time
            flickr_crawler.crawl(max_num=4000, text=class_name, sort='relevance', per_page=500,
                                 min_upload_date=TODAY - (i+1) * delta, max_upload_date=TODAY - i * delta,
                                 extras='url_n,url_z,original_format,path_alias')

        work_list.append(class_name)
        if i >= len(crawl_list) - 1:
            work_list.append('end')


if __name__ == '__main__':

    print(TODAY - 28 * delta)

    crawl_list = []
    for class_name in ['car', 'clock']:
        crawl_list.append(class_name)

    crawl(crawl_list, [])
