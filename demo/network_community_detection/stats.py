class StatsModule:
    @staticmethod
    def jaccard(nodeset1: set, nodeset2: set) -> float:
        intersection = len(list(nodeset1.intersection(nodeset2)))
        union = (len(nodeset1) + len(nodeset2)) - intersection
        return float(intersection) / union

    @staticmethod
    def standarize_sample_cluster_ordering(
        sample: dict,
        ref_sample: dict,
        communities: list,
        ref_communities: list,
    ) -> tuple[dict, list]:
        k = len(communities)
        rename = {}
        # Process communities with their random ordering
        for i in range(k):
            ref_comm = ref_communities[i]
            for j in range(k):
                comm = communities[j]
                if StatsModule.jaccard(comm, ref_comm) == 1:
                    intersection_nodes = comm.intersection(ref_comm)
                    rep = intersection_nodes.pop()
                    rename[j] = ref_sample[rep]

        renamed_sample = {node: rename[clus] for node, clus in sample.items()}

        renamed_communities = []
        for clus in range(k):
            comm = []
            for i in renamed_sample:
                if renamed_sample[i] == clus:
                    comm.append(i)
            renamed_communities.append(set(comm))

        return renamed_sample, renamed_communities
