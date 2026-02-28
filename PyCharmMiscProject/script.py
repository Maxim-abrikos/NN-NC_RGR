import sys
from PySide6.QtWidgets import QTextEdit
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QFileDialog
)
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def intra_cluster_distance(X, labels):
    unique_labels = np.unique(labels)
    total = 0
    count = 0

    for lab in unique_labels:
        if lab == -1:
            continue
        cluster_points = X[labels == lab]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        total += dists.sum()
        count += len(cluster_points)

    return total / count


def inter_cluster_distance(X, labels):
    unique_labels = [lab for lab in np.unique(labels) if lab != -1]
    centroids = []

    for lab in unique_labels:
        cluster_points = X[labels == lab]
        centroids.append(cluster_points.mean(axis=0))

    centroids = np.array(centroids)

    total = 0
    count = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            total += np.linalg.norm(centroids[i] - centroids[j])
            count += 1

    return total / count


def boundary_ratio(mesh, labels):
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    boundary_edges = 0
    total_edges = 0

    for f in faces:
        for i in range(3):
            v1 = f[i]
            v2 = f[(i + 1) % 3]
            total_edges += 1
            if labels[v1] != labels[v2]:
                boundary_edges += 1

    return boundary_edges / total_edges




# -------- PointNet-подобная модель --------
class PointNetSeg(nn.Module):
    def __init__(self, n_parts):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.mlp3 = nn.Linear(128, n_parts)  # предсказание сегмента

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return self.mlp3(x)  # N x n_parts


# -------- Viewer --------
class PyVistaWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plotter = QtInteractor(self)
        self.mesh = None

        layout = QVBoxLayout()
        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)

    def load_model(self, filename):
        self.mesh = pv.read(filename)
        self.mesh.compute_normals(inplace=True)
        self.show_mesh(color="lightgray")

    def show_mesh(self, face_colors=None, color=None):
        self.plotter.clear()
        if face_colors is not None:
            self.mesh.cell_data["colors"] = face_colors
            self.plotter.add_mesh(
                self.mesh,
                scalars="colors",
                rgb=True,
                smooth_shading=True
            )
        else:
            self.plotter.add_mesh(
                self.mesh,
                color=color,
                smooth_shading=True
            )
        self.plotter.reset_camera()
        self.plotter.render()

    # -------- Сегментация через PointNet --------
    def segment_model(self, n_parts, epochs=100, lr=0.01):
        if self.mesh is None:
            return

        vertices = self.mesh.points.astype(np.float32)
        scaler = StandardScaler()
        vertices_scaled = scaler.fit_transform(vertices)
        vertices_tensor = torch.tensor(vertices_scaled, dtype=torch.float32)

        # --- KMeans для эталонных сегментов (self-supervised) ---
        kmeans = KMeans(n_clusters=n_parts, n_init=10)
        labels = kmeans.fit_predict(vertices_scaled)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # --- Создаём PointNet ---
        model = PointNetSeg(n_parts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # --- Обучение на текущем объекте ---
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(vertices_tensor)
            loss = criterion(pred, labels_tensor)
            loss.backward()
            optimizer.step()

        # --- Предсказание сегментов ---
        # --- Предсказание сегментов через PointNet ---
        with torch.no_grad():
            pred_logits = model(vertices_tensor)  # N x n_parts
            pred_labels_pointnet = torch.argmax(pred_logits, dim=1).numpy()
            intra = intra_cluster_distance(vertices_scaled, pred_labels_pointnet)
            inter = inter_cluster_distance(vertices_scaled, pred_labels_pointnet)
            #sil = silhouette_score(vertices_scaled, pred_labels_pointnet)
            boundary = boundary_ratio(self.mesh, pred_labels_pointnet)
            compactness = inter / intra

            metrics_text = (
                f"Intra-cluster distance : {intra:.6f}\n"
                f"Inter-cluster distance : {inter:.6f}\n"
                #f"Silhouette score       : {sil:.6f}\n"
                f"Boundary ratio         : {boundary:.6f}\n"
                f"Compactness ratio      : {compactness:.6f}"
            )

            self.metrics_box.setText(metrics_text)

        # --- Сегментация граней через центр грани + спорные зоны ---
        faces = self.mesh.faces.reshape(-1, 4)[:, 1:4]
        face_centers = np.array([vertices_scaled[f].mean(axis=0) for f in faces])

        # расстояние от центра грани до центроидов KMeans
        dists_matrix = np.linalg.norm(face_centers[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2)

        # ближайший и второй ближайший центроид
        sorted_idx = np.argsort(dists_matrix, axis=1)
        nearest = sorted_idx[:, 0]
        second_nearest = sorted_idx[:, 1]

        # вершина слишком близка к двум центроидам → спорная зона
        threshold = 0.05  # можно подбирать
        mask_conflict = np.abs(dists_matrix[np.arange(len(face_centers)), nearest] -
                               dists_matrix[np.arange(len(face_centers)), second_nearest]) < threshold

        # финальные метки граней
        face_labels = nearest.copy()
        face_labels[mask_conflict] = -1  # спорные грани остаются пустыми

        # --- Цвета ---
        rng = np.random.default_rng(42)
        colors = rng.random((n_parts, 3))
        face_colors = np.zeros((faces.shape[0], 3))
        for i in range(faces.shape[0]):
            if face_labels[i] == -1:
                face_colors[i] = [0.7, 0.7, 0.7]  # серый для спорных зон
            else:
                face_colors[i] = colors[face_labels[i]]

        self.show_mesh(face_colors=face_colors)


# -------- Main Window --------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OBJ Segmenter (PointNet)")
        self.resize(1100, 800)

        self.viewer = PyVistaWidget()
        self.load_btn = QPushButton("Load OBJ")
        self.segment_btn = QPushButton("Segment")
        self.parts_spin = QSpinBox()
        self.parts_spin.setMinimum(2)
        self.parts_spin.setMaximum(20)
        self.parts_spin.setValue(6)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(QLabel("Number of parts:"))
        top_layout.addWidget(self.parts_spin)
        top_layout.addWidget(self.segment_btn)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.viewer)

        # --- Поле для вывода метрик ---
        self.metrics_box = QTextEdit()
        self.metrics_box.setReadOnly(True)
        self.metrics_box.setMaximumHeight(90)

        layout.addWidget(self.metrics_box)
        self.viewer.metrics_box = self.metrics_box

        self.setLayout(layout)

        self.load_btn.clicked.connect(self.load_model)
        self.segment_btn.clicked.connect(self.segment_model)

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open OBJ file", "", "OBJ Files (*.obj)")
        if filename:
            self.viewer.load_model(filename)

    def segment_model(self):
        n_parts = self.parts_spin.value()
        self.viewer.segment_model(n_parts)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())