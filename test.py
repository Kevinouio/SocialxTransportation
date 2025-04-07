import numpy as np

# Define the term-document matrix A (10x5)
A = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 0, 0]
], dtype=float)

# Define the query vector q (10-dimensional)
q = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)

# Compute the full SVD of A
U, s, Vh = np.linalg.svd(A, full_matrices=False)


L = np.array([0,0,0,1,1,0,1,0,0,0])

print(L @ L.T)

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


# Compute cosine similarities for each rank-k approximation
results = {}
for k in range(1, 6):
    # Truncated SVD components for rank k
    Uk = U[:, :k]
    Sk = np.diag(s[:k])
    Vhk = Vh[:k, :]
    print(f"\nSTEP {k} ")
    print(f"{Uk}\n")
    print(f"{Sk}\n")
    print(f"{Vhk}")


    # Document representation in the k-dimensional space
    Dk = Sk @ Vhk

    # Project query vector into the k-dimensional space
    qb_k = Uk.T @ q

    # Compute cosine similarities for each document (each column of Dk)
    cosines = []
    for j in range(5):
        doc_vec = Dk[:, j]
        cos_sim = cosine_similarity(qb_k, doc_vec)
        cosines.append(cos_sim)
    results[k] = np.array(cosines)

# Display the results
for k in range(1, 6):
    print(f"k = {k} cosines: {np.round(results[k], 4)}")

# For comparison, here are the original cosine similarities using A directly:
orig_cosines = []
for j in range(5):
    doc_vec = A[:, j]
    cos_sim = cosine_similarity(q, doc_vec)
    orig_cosines.append(cos_sim)
print("\nOriginal cosine similarities (full space):", np.round(orig_cosines, 4))
