import json
from operator import itemgetter

import networkx as nx


def load_graph(file, filter_not_loaded_nodes=True):
    graph = nx.DiGraph()

    url_to_info = dict()

    with open(file) as input_file:
        for line in input_file.readlines():
            node_info = json.loads(line)
            graph.add_node(node_info['url'])

            url_to_info[node_info['url']] = (node_info['title'], node_info['snippet'])

    with open(file) as input_file:
        for line in input_file.readlines():
            node_info = json.loads(line)
            node_url = node_info['url']
            for outlink in node_info['outlinks']:
                if not filter_not_loaded_nodes or outlink in url_to_info:
                    graph.add_edge(node_url, outlink)

    return graph, url_to_info


def analyze(G, url_to_info):
    print("Nodes count: {}".format(G.number_of_nodes()))
    print("Edges count: {}\n".format(G.number_of_edges()))

    def print_page(page_url, value):
        if page_url in url_to_info:
            page_attrs = url_to_info[page_url]
            print("\t{} {}\n\t{}\n\t{}\n".format(page_attrs[0], value, page_url, page_attrs[1]))
        else:
            print("\t{} {}\n\t{}\n\t{}\n".format('<No Title>', value, page_url, '<No snippet>'))

    def print_top10(ranks):
        graph_page_rank_sorted = sorted(list(ranks.items()), key=itemgetter(1), reverse=True)

        for page in graph_page_rank_sorted[:10]:
            print_page(page[0], page[1])

    alphas = [0.3, 0.5, 0.85, 0.95]

    for alpha in alphas:
        print("Top 10 by PageRank for alpha = {}\n".format(alpha))
        print_top10(nx.pagerank(G, alpha=alpha))

    hubs, authorities = nx.hits(G, max_iter=500)
    average = {url: (value + authorities[url]) / 2 for url, value in hubs.items()}
    print("Top 10 HITS: hubs\n")
    print_top10(hubs)
    print("Top 10 HITS: authorities\n")
    print_top10(authorities)
    print("Top 10 HITS: average\n")
    print_top10(average)


G1, url_to_info1 = load_graph('../scrapy_wiki/scrapy_results/wiki_links.json')
G2, url_to_info2 = load_graph('../scrapy_wiki/scrapy_results/wiki_links.json', False)

print("Results for loaded nodes only:")
analyze(G1, url_to_info1)

print("Results for all nodes:")
analyze(G2, url_to_info2)
