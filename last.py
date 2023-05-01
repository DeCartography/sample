import numpy as np
import random
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

class CrowdWorker:
    def __init__(self, name, humanity_score=0, staking_amount=0):
        self.name = name
        self.humanity_score = humanity_score
        self.staking_amount = staking_amount
        self.weight = humanity_score + np.sqrt(staking_amount) #quadratic staking
        self.choices = []

#depend on crowdworker's profile
workers = [
    CrowdWorker('CrowdWorkerA', 10, 0), 
    CrowdWorker('CrowdWorkerB', 45, 0.1), 
    CrowdWorker('CrowdWorkerC', 86, 0.05)
    ]

addresses = [
    '0x1f96522d02e1c11ca6e12cc51635fba12a8f7807', '0x241e82c79452f51fbfc89fac6d912e021db1a3b7',
    '0x09cabec1ead1c0ba254b09efb3ee13841712be14', '0xcda84cc75ec5c92a5dacabc13241256beceef964',
    '0x0f7ecf17f1abdeb5c53cf1dbba58b831c2a0d0c8', '0xc6581ce3a005e2801c1e0903281bbd318ec5b5c2',
    '0x60ea769c3b7b9c91bcf8d9c573db58f06e4efe12', '0x932348df588923ba3f1fd50593b22c4e2a287919',
    '0x11179c3cb11cd08ca22afb91e515257d5e7bf03c', '0x60ea769c3b7b9c91bcf8d9c573db58f06e4efe12',
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', '0x912fd21d7a69678227fe6d08c64222db41477ba0',
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', '0x61935cbdd02287b511119ddb11aeb42f1593b7ef',
    '0x951a280d2d94e5a374908a21531a5f37c3578ea3', '0x2a1530c4c41db0b0b2bb646cb5eb1a67b7158667',
    '0x15ae150d7dc03d3b635ee90b85219dbfe071ed35', '0xd924bdd6fa7fd3d0eb1337853a814a4263dcbfe8',
    '0x8387dbf85230975a26909c1240f6aea7eb45f9f3', '0x09cabec1ead1c0ba254b09efb3ee13841712be14',
    '0x932348df588923ba3f1fd50593b22c4e2a287919', '0x979ff11dcbf3ac66a4d15a7b3a5b306ccbbef4e9'
]

def worker_selection(worker, addresses):
    random_addresses = random.sample(addresses, 9)
    # Print the addresses for the worker to choose from.
    print(f"{worker.name}, please choose 3 addresses from the following list by typing the corresponding numbers (separated by space):")
    for i, address in enumerate(random_addresses):
        print(f"{i}: {address}")
    # Get the worker's choices.
    while True:
        choices = input().split()
        # Make sure the worker's choices are valid.
        if all(0 <= int(choice) < 9 for choice in choices) and len(choices) == 3:
            break
        else:
            print("Invalid input. Please choose 3 addresses by typing the corresponding numbers (0-8) separated by space.")
    # Add the worker's choices to their list of choices.
    chosen_addresses = [random_addresses[int(choice)] for choice in choices]
    worker.choices.extend(chosen_addresses)
    return chosen_addresses

n_sessions = 3 #number of sessions per worker
votes_matrix = np.zeros((len(addresses), len(addresses)))
worker_voting_power = {}

for worker in workers:
    for _ in range(n_sessions):
        chosen_addresses = worker_selection(worker, addresses)
        for address in chosen_addresses:
            votes_matrix[addresses.index(address), [addresses.index(chosen) for chosen in chosen_addresses]] += worker.weight
    print(f"Profile: Humanity Score - {worker.humanity_score}, Staking Amount - {worker.staking_amount}, Voting Weight - {worker.weight}")
    print(f"-------------")
    print(f"{worker.name} has finished their task. Please pass it on to the next person.")
    worker_voting_power[worker.name] = worker.weight * n_sessions

def peer_prediction(workers):
    correlation_scores = []
    for worker in workers:
        peers = [w for w in workers if w != worker]
        match_counts = [len(set(worker.choices).intersection(set(peer.choices))) for peer in peers]
        average_match_count = sum(match_counts) / len(match_counts)
        correlation_scores.append(average_match_count)
        print("Average match count for worker {} is {}".format(worker.name, average_match_count))
    return correlation_scores

def calculate_rewards(workers, correlation_scores, total_reward=100):
    weighted_scores = [math.sqrt(worker.weight) * correlation_scores[i] for i, worker in enumerate(workers)]
    total_weighted_scores = sum(weighted_scores)
    reward_ratios = [score / total_weighted_scores for score in weighted_scores]
    rewards = [ratio * total_reward for ratio in reward_ratios]
    print("Rewards for workers: {}".format(rewards))
    return rewards

def calculate_data(workers):
    data = {"Worker Name": [], "Total Voting Power": [], "Average Match Count": [], "Reward Distribution (%)": []}
    total_voting_power = {worker.name: worker.weight * n_sessions for worker in workers}

    correlation_scores = []
    for worker in workers:
        peers = [w for w in workers if w != worker]
        match_counts = [len(set(worker.choices).intersection(set(peer.choices))) for peer in peers]
        average_match_count = sum(match_counts) / len(match_counts)
        correlation_scores.append(average_match_count)
        print("Average match count for worker {} is {}".format(worker.name, average_match_count))

    total_reward = 100
    weighted_scores = [math.sqrt(worker.weight) * correlation_scores[i] for i, worker in enumerate(workers)]
    total_weighted_scores = sum(weighted_scores)
    reward_ratios = [score / total_weighted_scores for score in weighted_scores]
    rewards = [ratio * total_reward for ratio in reward_ratios]
    print("Rewards for workers: {}".format(rewards))

    for i, worker in enumerate(workers):
        data["Worker Name"].append(worker.name)
        data["Total Voting Power"].append(total_voting_power[worker.name])
        data["Average Match Count"].append(correlation_scores[i])
        data["Reward Distribution (%)"].append(rewards[i])
        
    return data

# Calculate data
data = calculate_data(workers)

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)


try:
    correlation_scores = peer_prediction(workers)
except:
    correlation_scores = np.zeros((len(addresses), len(addresses)))

calculate_rewards(workers, correlation_scores)

pca = PCA(n_components=2)
votes_pca = pca.fit_transform(votes_matrix)

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(votes_pca)

clustered_addresses = {i+1: [] for i in range(max(clusters)+1)}

for i, cluster in enumerate(clusters):
    clustered_addresses[cluster+1].append(addresses[i])

for cluster, addresses in clustered_addresses.items():
    print(f"Cluster {cluster}: {addresses}")

plt.figure(figsize=(10,10))
plt.scatter(votes_pca[:, 0], votes_pca[:, 1], c=clusters)

for i, txt in enumerate(addresses):
    plt.annotate(txt, (votes_pca[i, 0], votes_pca[i, 1]))

plt.show()