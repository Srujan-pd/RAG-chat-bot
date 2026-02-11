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
        self.contact_keywords = ['contact', 'about', 'company', 'support', 'help', 'reach', 'call', 'phone', 'email']

    def clean_url(self, url):
        url = url.split("#")[0].split("?")[0]
        return url.rstrip("/")

    def is_valid_url(self, url):
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False

        url_lower = url.lower()
        
        # ALWAYS include contact and about pages
        for keyword in self.contact_keywords:
            if keyword in url_lower:
                return True

        # Skip only non-content files and obvious junk
        skip_patterns = [
            r'\.(jpg|jpeg|png|gif|svg|pdf|zip|tar|gz|ico|css|js)$',
            r'/wp-(admin|includes|content/plugins|content/themes)',
            r'\?share=',
            r'#comment-',
            r'/tag/',
            r'/author/',
            r'/category/',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url_lower):
                return False
                
        return True

    def fetch(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        try:
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"❌ Fetch error for {url}: {e}")
            return None

    def extract_links(self, soup, current_url):
        links = set()
        for a in soup.find_all("a", href=True):
            href = a['href']
            if href.startswith('#') or href.startswith('javascript:'):
                continue
            full = urljoin(current_url, href)
            full = self.clean_url(full)
            if self.is_valid_url(full):
                links.add(full)
        return links

    def extract_text(self, soup):
        # Remove only script and style tags - preserve all content
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        # Special handling for contact information
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace but preserve line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Ensure phone numbers are preserved (they often have patterns like +1, 123-456-7890)
        return text

    def extract_contact_info(self, text, url):
        """Extract and highlight contact information"""
        contact_info = []
        
        # Phone number patterns
        phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\s*(?:ext|x|extension)?\s*\d*',
        ]
        
        # Email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info.extend([f"📞 Phone: {phone}" for phone in phones])
        
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info.extend([f"✉️ Email: {email}" for email in emails])
        
        return contact_info

    def crawl(self, max_pages=100):
        queue = [self.base_url]
        
        print(f"🚀 Crawling {self.base_url}")
        print(f"📊 Max pages: {max_pages}\n")

        while queue and len(self.visited) < max_pages:
            url = queue.pop(0)
            url = self.clean_url(url)

            if url in self.visited:
                continue

            try:
                print(f"📄 [{len(self.visited) + 1}/{max_pages}] {url}")
                
                res = self.fetch(url)
                if not res:
                    continue
                    
                soup = BeautifulSoup(res.text, "html.parser")

                self.visited.add(url)

                # Extract main content
                text = self.extract_text(soup)
                
                # Extract contact information specifically
                contact_info = self.extract_contact_info(text, url)
                
                # Only save pages with substantial content
                if len(text) > 200:
                    title = soup.title.string if soup.title else ""
                    page_data = {
                        "url": url,
                        "title": title.strip() if title else "",
                        "content": text,
                        "length": len(text),
                        "contact_info": contact_info,
                        "is_contact_page": any(kw in url.lower() for kw in self.contact_keywords)
                    }
                    self.pages.append(page_data)
                    
                    # Print contact info if found
                    if contact_info:
                        print(f"   ✅ Found contact info: {contact_info}")
                    
                    print(f"   ✅ Content length: {len(text)} chars")

                # Discover new links
                links = self.extract_links(soup, url)
                new_links = 0
                for link in links:
                    if link not in self.visited and link not in queue:
                        queue.append(link)
                        new_links += 1
                
                if new_links > 0:
                    print(f"   🔗 Found {new_links} new links")

                time.sleep(0.5)  # Polite delay

            except Exception as e:
                print(f"❌ Error on {url}: {e}")

        print(f"\n✅ Crawl complete!")
        print(f"   Visited: {len(self.visited)} pages")
        print(f"   Saved: {len(self.pages)} pages")
        
        # Summary of contact pages found
        contact_pages = [p for p in self.pages if p.get('is_contact_page')]
        print(f"   Contact pages: {len(contact_pages)}")
        
        return self.pages

    def save(self, filename="data/scraped_data.json"):
        os.makedirs("data", exist_ok=True)
        
        # Save full data
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, indent=2, ensure_ascii=False)
        
        # Also save a summary for quick verification
        summary = []
        for page in self.pages:
            summary.append({
                "url": page["url"],
                "title": page["title"][:100] if page["title"] else "",
                "length": page["length"],
                "has_contact": bool(page.get("contact_info")),
                "contact_info": page.get("contact_info", [])
            })
        
        summary_file = "data/scraped_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved full data to {filename}")
        print(f"📋 Saved summary to {summary_file}")


if __name__ == "__main__":
    scraper = PrimisWebsiteScraper()
    scraper.crawl(max_pages=100)
    scraper.save()
