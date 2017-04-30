import urllib
import threading

urls = []

for i in range(374236,374730):
    for j in range(218381,218714):
        urls.append('https://khms1.google.com/kh/v=722?x=' + str(i) + '&y=' + str(j) + '&z=19')

print len(urls)

downloaded = 0

def download_images(urls):
    global downloaded
    for url in urls:
        urllib.urlretrieve (url, "../../datasets/Delhi_imgs/" + url[36:42] + '_' + url[45:51] + ".jpg")

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

THREADS = 1

for urls_slice in list(chunks(urls, THREADS)):
    scrapper_thread = threading.Thread(target = download_images, args = (urls_slice,))
    scrapper_thread.start()