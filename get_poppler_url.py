import urllib.request
import re
import urllib.error

url = 'https://github.com/oschwartz10612/poppler-windows/releases'
try:
    req = urllib.request.Request('https://github.com/oschwartz10612/poppler-windows/releases/latest', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = re.findall(r'/oschwartz10612/poppler-windows/releases/download/[^\"]+\.zip', html)
    print("FOUND_LINKS:", links)
except Exception as e:
    print("ERROR:", e)
