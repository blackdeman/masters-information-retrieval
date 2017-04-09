# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapyWikiItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    snippet = scrapy.Field()
    outlinks = scrapy.Field()
    pass
