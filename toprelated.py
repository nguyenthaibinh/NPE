from annoy import AnnoyIndex


class ApproximateTopRelated(object):
    def __init__(self, features, metric='angular', treecount=20):
        print("index dimension:", features.shape[1])
        index = AnnoyIndex(features.shape[1], metric=metric)
        for i, row in enumerate(features):
            index.add_item(i, row)
        index.build(treecount)
        self.index = index

    def get_related(self, id, N=10):
        neighbours = self.index.get_nns_by_item(id, N)
        similarities = sorted(((other, 1 - self.index.get_distance(id, other))
                               for other in neighbours), key=lambda x: -x[1])
        sn = []
        ss = []
        for n, s in similarities:
            sn.append(n)
            ss.append(s)
        return neighbours, sn, ss
        # return
