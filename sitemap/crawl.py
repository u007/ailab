import aiohttp
import asyncio
import csv
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import sqlite3
from dotenv import load_dotenv
import os

load_dotenv()

# Define database connection and cursor
conn = sqlite3.connect('crawler.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS crawler (
    url TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    prefix TEXT,
    crawled INTEGER DEFAULT 0
)
''')
conn.commit()


async def remove_invalid_characters(input_string):
    try:
        # Try to encode the input string as UTF-8
        input_string.encode('utf-8')
        # If successful, return the original string
        return input_string
    except UnicodeEncodeError as e:
        # If encoding fails, identify the problematic characters and remove them
        problematic_characters = []
        for char in input_string:
            try:
                char.encode('utf-8')
            except UnicodeEncodeError:
                problematic_characters.append(char)

        # Remove problematic characters from the input string
        cleaned_string = ''.join(char for char in input_string if char not in problematic_characters)
        return cleaned_string


# Function to clean up the content
async def clean_content(content):
    content = re.sub(r'\r\n', '\n', content)  # Normalize Windows line endings
    content = re.sub(r'\r', '\n', content)  # Replace Mac line endings with Unix line endings
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove extra line breaks

    content = await remove_invalid_characters(content)
    return content


def replace_url(url, new_url):
    new_crawler = get_crawler_by_url(new_url)
    if new_crawler:
        print(f"replace_url {new_url} already exists")
        cursor.execute('update crawler set crawled=9 WHERE url = ?', (url,))
        conn.commit()
        return new_crawler

    cursor.execute('UPDATE crawler SET url = ? WHERE url = ?', (new_url, url))
    conn.commit()

    return get_crawler_by_url(new_url)


def update_crawler(url, title, content, crawled):
    cursor.execute('UPDATE crawler SET title = ?, content = ?, crawled = ? WHERE url = ?',
                   (title, content, crawled, url))
    conn.commit()


# Function to insert data into SQLite
def insert_into_db(url, title, content, prefix, crawled=0):
    row = get_crawler_by_url(url)
    if row:
        return
    print(f"insert_into_db {url}")
    cursor.execute(
        'INSERT OR IGNORE INTO crawler (url, title, content, prefix, crawled) VALUES (?, ?, ?, ?, ?)',
        (url, title, content, prefix, crawled))
    conn.commit()


def mark_as_failed(url, error):
    print(f"mark_as_failed {url} {error}")
    cursor.execute('UPDATE crawler SET crawled = 2 WHERE url = ?', (url,))
    conn.commit()


# Function to mark URL as crawled
def mark_as_crawled(url):
    cursor.execute('UPDATE crawler SET crawled = 1 WHERE url = ?', (url,))
    conn.commit()


# Function to get URLs not yet crawled
def get_uncrawled_urls():
    cursor.execute('SELECT crawled, url FROM crawler WHERE crawled = 0 order by url asc limit 5', ())
    res = []
    for row in cursor.fetchall():
        print(f"get_uncrawled_urls {row['url']} crawled: {row['crawled']}")
        res.append(row['url'])
    return res


def get_crawler_by_url(url):
    cursor.execute('SELECT crawled, url FROM crawler WHERE url = ?', (url,))
    return cursor.fetchone()


async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 404:
                print(f"Page not found: {url}")
                return None  # Skip processing if it's a 404 error
            elif response.status != 200:
                print(f"Unexpected status code {response.status} for URL: {url}")
                return None  # Skip processing for other non-200 status codes
            
            return await response.text()
    except Exception as e:
        print(f"Failed to fetch URL: {url}, Error: {e}")
        return None

async def crawl(url, prefix, session):
    url = url.strip()
    if not url.startswith(prefix):
        print(f"skip {url}")
        return

    new_url = url.replace('/ /', '/')
    if new_url != url:
        print(f"replace {url} with {new_url}")
        replace_url(url, new_url)
        url = new_url

    row = get_crawler_by_url(url)
    if row and row['crawled'] != 0:
        print(f"not pending crawled {url} crawled {row['crawled']} url: {row['url']}")
        return

    print(f"crawling {url}")
    response_text = await fetch_url(session, url)

    if response_text is None:
        mark_as_failed(url, "Failed to fetch URL")
        return

    soup = BeautifulSoup(response_text, 'html.parser')

    title = soup.find('title').get_text().strip() if soup.find('title') else 'No Title'
    content = await clean_content(soup.find('body').get_text(separator='\n').strip() if soup.find('body') else 'No Content')

    print(f"crawl {url} {title} len: {len(content)}")

    for link in soup.find_all('a', href=True):
        linked_url = urljoin(url, link['href'])
        linked_url = urlparse(linked_url).geturl()

        row = get_crawler_by_url(linked_url)
        if row and row['crawled'] == 1:
            continue
        if linked_url.startswith(prefix):
            insert_into_db(linked_url, '', '', prefix)

    update_crawler(url, title, content, 1)

def export_to_csv(query, write_file):
    # Execute the query
    cursor.execute(query)

    with open(write_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['URL', 'Title', 'Content'])  # Write headers

        # Fetch and write rows one by one
        for row in cursor:
            writer.writerow(row)

async def main(site):
    uncrawled_urls = get_uncrawled_urls()
    if not uncrawled_urls:
        print(f"new crawl {site}")
        crawl(site, site)
        uncrawled_urls = get_uncrawled_urls()

    print(f"uncrawled_urls {len(uncrawled_urls)}")

    while uncrawled_urls:
        async with aiohttp.ClientSession() as session:
            tasks = [crawl(url, site, session) for url in uncrawled_urls]
            await asyncio.gather(*tasks)
        
        uncrawled_urls = get_uncrawled_urls()

    export_to_csv("select * from crawler where crawled=1 order by url asc", "result.csv")
    conn.close()


if __name__ == "__main__":
    url = os.getenv('URL')
    asyncio.run(main(url))
