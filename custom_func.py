from datetime import datetime
import os

HTML_DIR = os.path.join(os.getcwd(),'html')

def write_html(html:str):
    title = "html_page"
    with open (HTML_DIR+"/{}-{:%d%m%Y_%H%M%S}.html".format(title, datetime.now()),'w') as f:
        f.write(html)

def main():
    if not os.path.exists(HTML_DIR):
        os.mkdir(HTML_DIR)


if __name__ == '__main__':
    main()
