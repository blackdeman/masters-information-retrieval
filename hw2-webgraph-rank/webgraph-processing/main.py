import networkx as nx
import json
from operator import itemgetter


def load_graph(file):
    graph = nx.DiGraph()

    url_to_info = dict()

    with open(file) as input_file:
        for line in input_file.readlines():
            node_info = json.loads(line)
            graph.add_node(node_info['url'])

            url_to_info[node_info['url']] = (
                node_info['title'][0] if len(node_info['title']) > 0 else '', node_info['snippet'])

    with open(file) as input_file:
        for line in input_file.readlines():
            node_info = json.loads(line)
            node_url = node_info['url']
            for outlink in node_info['outlinks']:
                if outlink in url_to_info:
                    graph.add_edge(node_url, outlink)

    return graph, url_to_info


G, url_to_info = load_graph('../scrapy_wiki/scrapy_results/wiki_links_12286.json')

print("Nodes count: {}".format(G.number_of_nodes()))
print("Edges count: {}".format(G.number_of_edges()))


def print_page(page_url, value):
    page_attrs = url_to_info[page_url]
    print("\t{} {}\n\t{}\n\t{}\n".format(page_attrs[0], value, page_url, page_attrs[1]))


def print_top10(ranks):
    graph_page_rank_sorted = sorted(list(ranks.items()), key=itemgetter(1), reverse=True)

    for page in graph_page_rank_sorted[:10]:
        print_page(page[0], page[1])


alphas = [0.3, 0.5, 0.85, 0.95]

for alpha in alphas:
    print("Top 10 by PageRank for alpha = {}\n".format(alpha))
    print_top10(nx.pagerank(G, alpha=alpha))

hubs, authorities = nx.hits(G)
average = {url: (value + authorities[url]) / 2  for url, value in hubs.items()}
print("Top 10 HITS: hubs\n")
print_top10(hubs)
print("Top 10 HITS: authorities\n")
print_top10(authorities)
print("Top 10 HITS: average\n")
print_top10(average)