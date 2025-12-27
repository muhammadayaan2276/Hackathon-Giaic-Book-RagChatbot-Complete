from bs4 import BeautifulSoup
import re

class Extractor:
    def __init__(self):
        pass

    def extract_content(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove common boilerplate elements
        for selector in ['nav', 'footer', 'aside', '.sidebar', '.navbar', '.table-of-contents', '.doc-navigation']:
            for tag in soup.select(selector):
                tag.decompose()

        # Extract title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Extract main content
        main_content_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'code', 'pre', 'blockquote'])
        
        content_parts = []
        for tag in main_content_tags:
            if tag.name.startswith('h'):
                content_parts.append(f"\n{tag.get_text(strip=True)}\n")
            else:
                content_parts.append(tag.get_text(separator=' ', strip=True))
        
        extracted_text = "\n".join(content_parts)

        # Normalize and clean text
        extracted_text = re.sub(r'\n\s*\n', '\n\n', extracted_text) # Remove excessive blank lines
        extracted_text = re.sub(r'[ \t]+', ' ', extracted_text)     # Replace multiple spaces/tabs with single space
        extracted_text = extracted_text.strip()                     # Remove leading/trailing whitespace

        return {"title": title, "content": extracted_text}
