import numpy as np
from sklearn.cluster import MeanShift

def successive_halving_budget(self, total_budget, num_stages):
    budgets = []
    denominator = sum([2 ** (stage - 1) for stage in range(1, num_stages + 1)])
    for stage in range(1, num_stages + 1):
        budget = total_budget * (2 ** (num_stages - stage)) / denominator
        budgets.append(budget)
    return budgets


def select_fidelity_for_stage3(self, optimized_fidelity_pop, stage_index, num_stages):

    fronts = self.multi_fidelity_optimizer.fast_non_dominated_sort(optimized_fidelity_pop)
    first_front = fronts[0]

    corrs = np.array([x[1] for x in first_front]).reshape(-1, 1)

    corrs = np.array([0.12, 0.15, 0.19, 0.35, 0.39, 0.40, 0.77, 0.86, 0.81]).reshape(-1, 1)

    mean_shift = MeanShift()
    mean_shift.fit(corrs)
    labels = mean_shift.labels_

    cluster_centers = mean_shift.cluster_centers_.flatten()

    print("Fidelity Values:", corrs.flatten())
    print("Cluster Centers:", cluster_centers)

    clusters = {}
    for label in np.unique(labels):
        clusters[label] = [first_front[i] for i in range(len(labels)) if labels[i] == label]
    sorted_centers = sorted(cluster_centers)

    if stage_index < len(sorted_centers):
        selected_center = sorted_centers[stage_index]
    else:
        selected_center = sorted_centers[-1]

    current_stage_fidelities = clusters[np.where(cluster_centers == selected_center)[0][0]]
    selected_fidelity_tuple = min(current_stage_fidelities, key=lambda x: x[2])

    return selected_fidelity_tuple[0], selected_fidelity_tuple[1]


def select_fidelity_for_stages2(self, optimized_fidelity_pop, num_stages):
    fronts = self.multi_fidelity_optimizer.fast_non_dominated_sort(optimized_fidelity_pop)
    first_front = fronts[0]
    first_front = [ind for ind in first_front if ind[1] > 0]
    first_front.sort(key=lambda x: x[1])

    if not first_front:
        raise ValueError("No positive fidelity values found in the first front.")

    num_fidelities = len(first_front)
    if num_fidelities <= num_stages:
        selected_fidelities = [first_front[min(i, num_fidelities - 1)] for i in range(num_stages)]
    else:

        clusters = [[f] for f in first_front]
        centroids = [f[1] for f in first_front]

        while len(clusters) > num_stages:
            min_distance = float('inf')
            to_merge = (0, 1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = abs(centroids[i] - centroids[j])
                    if distance < min_distance:
                        min_distance = distance
                        to_merge = (i, j)

            clusters[to_merge[0]].extend(clusters[to_merge[1]])
            del clusters[to_merge[1]]
            centroids[to_merge[0]] = self.calculate_centroid(clusters[to_merge[0]])
            del centroids[to_merge[1]]
        selected_fidelities = [min(cluster, key=lambda x: x[2]) for cluster in clusters]

    return [(f[0], f[1]) for f in selected_fidelities]


def select_fidelity_for_stage(self, optimized_fidelity_pop, stage_index, num_stages):

    fronts = self.multi_fidelity_optimizer.fast_non_dominated_sort(optimized_fidelity_pop)
    first_front = fronts[0]

    first_front.sort(key=lambda x: x[1])

    num_fidelities = len(first_front)
    if num_fidelities < num_stages:
        actual_num_stages = num_fidelities
    else:
        actual_num_stages = num_stages

    step = num_fidelities / actual_num_stages

    if stage_index < actual_num_stages:
        start_index = int(stage_index * step)
        end_index = int((stage_index + 1) * step)
        current_stage_fidelities = first_front[start_index: end_index]
        selected_fidelity_tuple = min(current_stage_fidelities, key=lambda x: x[2])
    else:

        selected_fidelity_tuple = first_front[-1]

    return selected_fidelity_tuple[0], selected_fidelity_tuple[1]