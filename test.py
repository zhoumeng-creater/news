# pip install firecrawl-py
from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-2fe3b666c0a549fd978c174a96c40c4c")

# Scrape a website:
app.scrape('firecrawl.dev')

