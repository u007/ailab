import requests
from bs4 import BeautifulSoup
import csv
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

# Function to clean up the content
def clean_content(content):
    content = re.sub(r'\r\n', '\n', content)  # Normalize Windows line endings
    content = re.sub(r'\r', '\n', content)  # Replace Mac line endings with Unix line endings
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove extra line breaks
    return content

# Function to insert data into SQLite
def insert_into_db(url, title, content, prefix, crawled=0):
    row = get_crawler_by_url(url)
    if row:
        return
    print(f"insert_into_db {url}")
    cursor.execute('INSERT OR IGNORE INTO crawler (url, title, content, prefix, crawled) VALUES (?, ?, ?, ?, ?)',
                   (url, title, content, prefix, crawled))
    conn.commit()

# Function to mark URL as crawled
def mark_as_crawled(url):
    cursor.execute('UPDATE crawler SET crawled = 1 WHERE url = ?', (url,))
    conn.commit()

# Function to get URLs not yet crawled
def get_uncrawled_urls(prefix):
    cursor.execute('SELECT url FROM crawler WHERE crawled = 0 AND prefix = ?', (prefix,))
    return [row['url'] for row in cursor.fetchall()]

def get_crawler_by_url(url):
    cursor.execute('SELECT * FROM crawler WHERE url = ?', (url,))
    return cursor.fetchone()

# Function to crawl a webpage
def crawl(url, prefix):
    url = url.strip()
    if not url.startswith(prefix):
        print(f"skip {url}")
        return
    
    row = get_crawler_by_url(url)
    # print(f"row {row} crawled? %s" % row['crawled'] if row else "row is None")
    if row and row['crawled'] == 1:
        return

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find('title').get_text().strip() if soup.find('title') else 'No Title'
    content = clean_content(soup.find('body').get_text(separator='\n').strip() if soup.find('body') else 'No Content')

    # Store the crawled information in the SQLite database
    insert_into_db(url, title, content, prefix)

    # Recursively crawl the found links
    for link in soup.find_all('a', href=True):
        linked_url = urljoin(url, link['href'])
        linked_url = urlparse(linked_url).geturl()


        row = get_crawler_by_url(url)
        # print(f"row {row}") 
        if row and row['crawled'] == 1:
            continue
        if linked_url.startswith(prefix):
            insert_into_db(linked_url, '', '', prefix)
    # print(f"links? {soup.find_all('a', href=True)}")
    
    mark_as_crawled(url)

# Function to export data to CSV
def export_to_csv():
    write_file = "result.csv"
    cursor.execute('SELECT url, title, content FROM crawler ORDER BY url')
    rows = cursor.fetchall()

    with open(write_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['URL', 'Title', 'Content'])  # Write headers
        writer.writerows(rows)  # Write data rows

# Main function
def main(site):
    uncrawled_urls = get_uncrawled_urls(site)
    if not uncrawled_urls:  # Start with the main site if no URLs in DB
        crawl(site, site)
        
    uncrawled_urls = get_uncrawled_urls(site)
    while len(uncrawled_urls) > 0:
        for uncrawled_url in uncrawled_urls:  # Resume crawling with uncrawled URLs
            crawl(uncrawled_url, site)
        
        uncrawled_urls = get_uncrawled_urls(site)

    export_to_csv()  # Export the data to CSV

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    url = os.getenv('URL')
    main(url)
