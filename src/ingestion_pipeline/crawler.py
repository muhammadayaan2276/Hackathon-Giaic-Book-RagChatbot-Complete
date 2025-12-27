import requests
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Crawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_urls = set()
        self.internal_urls = set()
        self.skipped_urls = set()
        self.failed_urls = set()

    def crawl(self):
        logging.info(f"Starting crawl from: {self.base_url}")
        self._crawl_page(self.base_url)
        logging.info(f"Crawl completed. Found {len(self.internal_urls)} internal Docusaurus documentation pages.")
        logging.info(f"Skipped {len(self.skipped_urls)} URLs.")
        logging.info(f"Failed to crawl {len(self.failed_urls)} URLs.")
        return list(self.internal_urls)

    def _crawl_page(self, url):
        if url in self.visited_urls:
            return
        
        logging.info(f"Crawling: {url}")
        self.visited_urls.add(url)

        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            logging.error(f"Error crawling {url}: {e}")
            self.failed_urls.add(url)
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            absolute_link = urljoin(url, link['href'])
            if self._is_internal_docusaurus_doc(absolute_link):
                if absolute_link not in self.internal_urls:
                    self.internal_urls.add(absolute_link)
                    self._crawl_page(absolute_link)
            else:
                self.skipped_urls.add(absolute_link)
                logging.debug(f"Skipping non-doc URL: {absolute_link}")

    def _is_internal_docusaurus_doc(self, url):
        # Refined filtering for Docusaurus documentation pages
        parsed_url = urlparse(url)
        base_parsed_url = urlparse(self.base_url)

        # Must be on the same domain or a subdomain (e.g. docs.example.com for example.com)
        if not (parsed_url.netloc == base_parsed_url.netloc or parsed_url.netloc.endswith('.' + base_parsed_url.netloc)):
            return False
        
        # Must start with the base path
        if not parsed_url.path.startswith(base_parsed_url.path):
            return False

        # Must contain /docs/ segment
        if "/docs/" not in parsed_url.path:
            return False
        
        # Exclude common asset extensions and anchors
        excluded_extensions = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".json", ".xml", ".zip", ".pdf")
        if parsed_url.path.endswith(excluded_extensions) or parsed_url.fragment:
            return False
        
        return True

