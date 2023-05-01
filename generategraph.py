import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

def worker_selection(addresses):
    random_addresses = random.sample(addresses, 9)
    print("Please choose 3 addresses from the following list by typing the corresponding numbers (separated by space):")
    for i, address in enumerate(random_addresses):
        print(f"{i}: {address}")
    while True:
        choices = input().split()
        if all(0 <= int(choice) < 9 for choice in choices) and len(choices) == 3:
            break
        else:
            print("Invalid input. Please choose 3 addresses by typing the corresponding numbers (0-8) separated by space.")
    chosen_addresses = [random_addresses[int(choice)] for choice in choices]
    return chosen_addresses

n_workers = 3
n_sessions = 3
votes_matrix = np.zeros((len(addresses), len(addresses)))

for worker in range(n_workers):
    for _ in range(n_sessions):
        chosen_addresses = worker_selection(addresses)
        for address in chosen_addresses:
            votes_matrix[addresses.index(address), [addresses.index(chosen) for chosen in chosen_addresses]] += 1
    print(f"Worker {worker+1} has finished their task. Please pass it on to the next person.")

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
