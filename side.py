import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
import seaborn as sns
import random


def euclidean(point, data):
    # Khoảng cách Euclidean giữa 2 điểm
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=1000):
        self.prev_centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):

        # Khởi tạo ngẫu nhiên các centroids sử dụng công thức "k-means++"
        # Chọn ngẫu nhiên 1 điểm dữ liệu làm centroid đầu tiên
        self.centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters - 1):
            # Tính tổng khoảng cách từ các điểm dữ liệu đến các centroids khác
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Chuẩn hóa tổng khoảng cách
            dists /= np.sum(dists)
            # Chọn ngẫu nhiên với xác suất bằng khoảng cách từ điểm dữ liệu đến các centroids
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]
            self.centroids += [X_train[new_centroid_idx]]

        # Công thức chọn centroid cơ bản, kém hiệu quả hơn
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Lặp và điều chỉnh vị trí centroids cho đến khi hội tụ (số điểm thuộc 1 cluster không thay đổi) hoặc đạt số lần lặp tối đa
        iteration = 0
        while self.prev_centroids is not None and np.not_equal(self.centroids,
                                                               self.prev_centroids).any() and iteration < self.max_iter:
            # Sắp xếp từng datapoint vào cluster gần nhất tương ứng
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Thay đổi centroids ứng với trung bình cộng của các datapoint trong cluster
            self.prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            # for i, centroid in enumerate(self.centroids):
            #     if np.isnan(centroid).any():  # Nếu một centroids không có datapoint nào thuộc về nó
            #         self.centroids[i] = self.prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs


centers = 6
# Tạo dữ liệu
df = pd.read_csv('housing.csv')
X_train = df.loc[:, ['Latitude', 'Longitude']].values
print("X_train:", X_train)

# Scale data (Tập data này dã đuợc scale rồi nên không sử dụng)
# X_train = StandardScaler().fit_transform(X_train)

# K-means
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

# Biểu diễn kết quả
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=classification,
                palette="deep",
                legend=None
                )

plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '*',
         markersize=10,
         color='black'
         )
plt.title("k-means")
plt.show()
