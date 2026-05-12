import urllib.request
import re
import urllib.error

url = 'https://github.com/UB-Mannheim/tesseract/wiki'
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = re.findall(r'href="([^"]+tesseract-ocr-w64-setup[^"]+\.exe)"', html)
    print("FOUND_LINKS:", links)
except Exception as e:
    print("ERROR:", e)
