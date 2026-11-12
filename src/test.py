from tinyfish import TinyFish

client = TinyFish()

# Step 1: Search
results = client.search.query(query="latest EU politics news", domain_type="news")

top_urls = [r.url for r in results.results[:2]]

# Step 2: Fetch full content
pages = client.fetch.get_contents(urls=[top_urls[0]], format="markdown")
# for page in pages.results:
#     print(f"## {page.title}")
#     print(page.text, "\n")

page = pages.results[0]
# print(f"Title: {page.title}")
# print(page.text)



def fetch_content(text, max_tries=5, min_chars=300):
    # Web search by query 'text'
    response = client.search.query(query=text, domain_type="news")
    top_urls = [r.url for r in response.results[:max_tries]]

    # Try to fetch in list of URLs retrieved
    for i, url in enumerate(top_urls, start=1):
        link = url
        print(f"[{i}/{len(top_urls)}] Fetching URL: {link}")

        # fetche content from i-th url in the list
        fetched = client.fetch.get_contents(urls=[url], format="markdown").results[0]
        content = fetched.text
        print(f"Title: {fetched.title}")
        print(fetched.text)

        if content and len(content.strip()) >= min_chars:
            print("  OK")
            return content           # stop at the first success

        print("  too little text, skipping")

    return None, None # nothing works


fetch_content("latest F1 racing news", 10, 500)