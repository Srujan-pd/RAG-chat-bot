import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import os
import re

class PrimisWebsiteScraper:
    def __init__(self, base_url="https://primisdigital.com/"):
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(self.base_url).netloc
        self.visited = set()
        self.pages = []

    def clean_url(self, url):
        url = url.split("#")[0].split("?")[0]
        return url.rstrip("/")

    def is_valid_url(self, url):
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False

        # Skip junk
        skip_patterns = [
            r'\.(jpg|jpeg|png|gif|svg|pdf)$',
            r'/tag/',
            r'/author/',
            r'/wp-',
        ]
        for pattern in skip_patterns:
            if re.search(pattern, url):
                return False
        return True

    def fetch(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        return requests.get(url, timeout=15, headers=headers)

    def extract_links(self, soup, current_url):
        links = set()
        for a in soup.find_all("a", href=True):
            full = urljoin(current_url, a["href"])
            full = self.clean_url(full)
            if self.is_valid_url(full):
                links.add(full)
        return links

    def extract_text(self, soup):
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return text

    def crawl(self, max_pages=80):
        queue = [self.base_url]

        print(f"🚀 Crawling {self.base_url}\n")

        while queue and len(self.visited) < max_pages:
            url = queue.pop(0)
            url = self.clean_url(url)

            if url in self.visited:
                continue

            try:
                print(f"📄 {url}")
                res = self.fetch(url)
                soup = BeautifulSoup(res.text, "html.parser")

                self.visited.add(url)

                # Save content
                text = self.extract_text(soup)
                if len(text) > 500:
                    title = soup.title.string if soup.title else ""
                    self.pages.append({
                        "url": url,
                        "title": title.strip(),
                        "content": text,
                        "length": len(text),
                    })

                # Discover new links
                links = self.extract_links(soup, url)
                for link in links:
                    if link not in self.visited:
                        queue.append(link)

                time.sleep(0.4)

            except Exception as e:
                print(f"❌ {e}")

        print(f"\n✅ Crawled: {len(self.visited)} pages")
        print(f"✅ Saved: {len(self.pages)} pages")
        return self.pages

    def save(self, filename="data/scraped_data.json"):
        os.makedirs("data", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filename}")


if __name__ == "__main__":
    scraper = PrimisWebsiteScraper()
    scraper.crawl()
    scraper.save()

