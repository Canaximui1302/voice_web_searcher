from tinyfish import TinyFish

client = TinyFish()

# Step 1: Search
results = client.search.query(query="latest EU politics news", domain_type="news")

top_urls = [r.url for r in results.results[:2]]

# Step 2: Fetch full content
pages = client.fetch.get_contents(urls=top_urls, format="markdown")
for page in pages.results:
    print(f"## {page.title}")
    print(page.text, "\n")