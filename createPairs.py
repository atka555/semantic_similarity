import json
import glob
from platform import node
import random
from pprint import pprint
import csv

maps = glob.glob("../corpora/polish/*.json")


def aduPairs(edgePairs, nodesById):
    """
    aduPairs create list of ADU (Argumentative Discourse Units)  pairs containing connected conclusion and premise [[conclusion, premise]]
    """
    aduPair = []
    for pair in edgePairs.values():
        for p in pair["premises"]:
            aduPair.append(
                [
                    nodesById[pair["conclusion"]["toID"]]["text"],
                    nodesById[p["fromID"]]["text"],
                ]
            )
    return aduPair


def conclusionPremiseDict(premises, conclusions):
    """
    conclusionPremiseDict Create dictionary of pairs with an identifier with the following form: {id: {"conclusion": , "premises":[]}}
    """
    pairs = {}
    for i, x in enumerate(conclusions):
        pairs[i] = {"conclusion": x, "premises": []}
        id_to = x["fromID"]
        for p in premises:
            if p["toID"] == id_to:
                pairs[i]["premises"].append(p)

    return pairs


def pairs(map):
    """
    pairs creates conclusion - premise pairs for one map
    """

    with open(map) as f:
        data = json.loads(f.read())
    # Creating nodesById dictionary which has nodeID as key and whole node as value for more efficient data extraction.
    nodesById = {}
    for _, node in enumerate(data["nodes"]):
        nodesById[node["nodeID"]] = node

    # Premises are nodes that have ingoing edges that are type 'RA' and outgoing edges that are type 'I'.
    premises = [
        x
        for x in data["edges"]
        if nodesById[x["fromID"]]["type"] == "I"
        and nodesById[x["toID"]]["type"] == "RA"
    ]

    # Conclusions are nodes that have ingoing edges that are type 'I' and outgoing edges that are type 'RA'.
    conclusions = [
        x
        for x in data["edges"]
        if nodesById[x["toID"]]["type"] == "I"
        and nodesById[x["fromID"]]["type"] == "RA"
    ]
    edgePairs = conclusionPremiseDict(premises, conclusions)
    adus = aduPairs(edgePairs, nodesById)
    return adus, conclusions, premises, nodesById


#
# comb makes combination of conclusions and premises lists and returns list of pairs that are not conclusion-premise pairs
#
def comb(conclusions, premises, l, nodesById):
    combList = [(x, y) for x in conclusions for y in premises]
    smallCombList = []
    for _ in range(l):
        p = random.choice(combList)
        smallCombList.append(
            [nodesById[p[0]["toID"]]["text"], nodesById[p[1]["fromID"]]["text"]]
        )
    return smallCombList


truePairs = []
conclusions = []
premises = []
nodesById = {}

for m in maps:
    adus, c, p, n = pairs(m)
    truePairs.extend(adus)
    conclusions.extend(c)
    premises.extend(p)
    nodesById = {**nodesById, **n}

falsePairs = comb(conclusions, premises, len(truePairs), nodesById)

"""
    randomPairs makes combination of ADU's from whole corpus and returns list of pairs that are not realted and are not premises or conclusions
"""


def selectNodeText(p, nodesById, param):
    elements = ["I", "L", "YA", "TA", "RA", "Default Inference"]
    if nodesById[p[param]["toID"]]["text"] not in elements:
        return nodesById[p[param]["toID"]]["text"]
    else:
        return nodesById[p[param]["fromID"]]["text"]


def randomPairsComb(nodes, l, nodesById):
    randList = [(x, y) for x in conclusions for y in premises]
    randList.extend([(x, y) for x in premises for y in premises])
    randList.extend([(x, y) for x in premises for y in conclusions])
    randList.extend([(x, y) for x in conclusions for y in conclusions])

    smallRandList = []
    for _ in range(l):
        p = random.choice(randList)
        smallRandList.append(
            [selectNodeText(p, nodesById, 0), selectNodeText(p, nodesById, 1)]
        )
    return smallRandList


randomPairs = randomPairsComb(nodesById, len(truePairs), nodesById)

print(truePairs)

with open("polishTruePairs", "w") as f:
    writer = csv.writer(f)
    writer.writerows(truePairs)


f = open('polishFalsePairs', 'w')
writer = csv.writer(f)
writer.writerows(falsePairs)


f = open('polishRandomPairs', 'w')
writer = csv.writer(f)
writer.writerows(randomPairs)


