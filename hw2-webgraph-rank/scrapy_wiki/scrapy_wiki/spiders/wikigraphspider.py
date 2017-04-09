# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from scrapy_wiki.items import ScrapyWikiItem


class WikiGraphSpider(CrawlSpider):
    name = "wikigraph"
    allowed_domains = ["en.wikipedia.org"]
    start_urls = ['https://en.wikipedia.org/wiki/Wiki',
                  'https://en.wikipedia.org/wiki/Facebook',
                  'https://en.wikipedia.org/wiki/United_States',
                  'https://en.wikipedia.org/wiki/Philosophy']

    link_extractor = LinkExtractor(
        # filter special wiki pages
        deny='https://en\.wikipedia\.org/wiki/'
             '((File|Talk|Category|Portal|Special|Template|Template_talk|Wikipedia|Help|Draft):.*|Main_Page)',
        # extract links from main content
        restrict_xpaths='//div[@id="mw-content-text"]/*/a'
    )

    rules = (
        Rule(
            link_extractor=link_extractor,
            callback='parse_item',
            follow=True,
            process_links='process_links'
        ),
    )

    # extract only 100 first links
    def process_links(self, links):
        return links[:100]

    def parse_start_url(self, response):
        return self.parse_item(response)

    def parse_item(self, response):
        item = ScrapyWikiItem()
        item['url'] = response.url
        item['title'] = response.xpath('//h1[@id="firstHeading"]/text()').extract()

        item['snippet'] = BeautifulSoup(
            response.xpath('//div[@id="mw-content-text"]/p[not(descendant::span[@id="coordinates"])][1]').extract_first(),
            "lxml"
        ).text[:255] + "..."

        outlinks = set()
        for lnk in self.link_extractor.extract_links(response):
            lnk_url = lnk.url
            if lnk_url not in outlinks:
                outlinks.add(lnk_url)

        # ignore self links
        if response.url in outlinks: outlinks.remove(response.url)

        item['outlinks'] = outlinks

        return item
