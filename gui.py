import cv2
import os
import sys
import numpy as np
import subprocess
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QTextEdit, QSplitter, QComboBox, QHBoxLayout,
    QMessageBox, QToolBar, QInputDialog
)
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import Qt, QTimer
from calibration_module import calibrate_camera, save_calibration
from processing import run_processing

def find_available_cameras(max_index=10):
    available = []
    for i in range(1, max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available

class ImageViewer(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Aperçu Image")
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.display_image(image)
        self.resize(800, 600)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

class StereoVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Stéréo - Reconstruction 3D")
        self.resize(1400, 900)
        self.setStyleSheet("background-color: #D3D3D3;")
        self.capture_folder = os.path.join(os.getcwd(), "captures")
        os.makedirs(self.capture_folder, exist_ok=True)
        self.toolbar = QToolBar("Fonctions")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.statusBar().showMessage("Prêt")
        self.calibration_data = None
        self.calibration_loaded = False
        self.cam_selector_1 = QComboBox()
        self.cam_selector_2 = QComboBox()
        available_cams = find_available_cameras()
        for i in available_cams:
            self.cam_selector_1.addItem(f"Caméra {i}", i)
            self.cam_selector_2.addItem(f"Caméra {i}", i)
        self.init_ui()
        self.load_calibration_if_exists()
        if len(available_cams) < 2:
            self.log_message("❌ Deux caméras USB non détectées.")
            QMessageBox.critical(self, "Erreur", "Deux caméras USB sont requises.")
        else:
            self.init_cameras(available_cams[0], available_cams[1])

    def init_ui(self):
        self.left_image_label = QLabel(self)
        self.right_image_label = QLabel(self)
        for label in [self.left_image_label, self.right_image_label]:
            label.setFixedSize(640, 480)
            label.setStyleSheet("border: 2px solid black; background: #A9A9A9;")
            label.mousePressEvent = self.show_fullscreen_image
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background: black; color: white; font-family: Consolas; padding: 5px;")
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Caméra Gauche :"))
        cam_layout.addWidget(self.cam_selector_1)
        cam_layout.addWidget(QLabel("Caméra Droite :"))
        cam_layout.addWidget(self.cam_selector_2)
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_image_label)
        image_layout.addWidget(self.right_image_label)
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        top_layout.addLayout(cam_layout)
        top_layout.addLayout(image_layout)
        top_widget.setLayout(top_layout)
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Console de Debug :"))
        log_layout.addWidget(self.log_console)
        bottom_widget = QWidget()
        bottom_widget.setLayout(log_layout)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([700, 200])
        self.setCentralWidget(splitter)
        self.add_toolbar_buttons()

    def init_cameras(self, cam1_index, cam2_index):
        self.cap1 = cv2.VideoCapture(cam1_index)
        self.cap2 = cv2.VideoCapture(cam2_index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_views)
        self.timer.start(30)
        self.log_message(f"✅ Caméras {cam1_index} et {cam2_index} initialisées.")

    def update_camera_views(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        if ret1:
            self.left_image = frame1
            self.display_image(frame1, self.left_image_label)
        if ret2:
            self.right_image = frame2
            self.display_image(frame2, self.right_image_label)

    def load_calibration_if_exists(self):
        try:
            data = np.load("calibration_result.npz")
            self.K = data["K"]
            self.dist = data["dist"]
            self.calibration_loaded = True
            self.log_message("✅ Paramètres de calibration chargés.")
        except:
            self.calibration_loaded = False
            self.log_message("⚠️ Paramètres de calibration non trouvés.")

    def add_toolbar_buttons(self):
        def add_button(name, callback):
            btn = QAction(name, self)
            btn.triggered.connect(callback)
            self.toolbar.addAction(btn)
        add_button("📂 Charger Image Gauche", self.load_left_image)
        add_button("📂 Charger Image Droite", self.load_right_image)
        add_button("📸 Capturer", self.capture_images)
        add_button("📂 Charger dossier échiquiers", self.load_chessboard_folder_for_calibration)
        add_button("🔧 Aide", self.calibrate_camera_from_folder)
        add_button("⚙️ Traitement", self.process_stereo_vision)
        add_button("🌐 Ouvrir nuage 3D", self.open_point_cloud_file)

    def open_point_cloud_file(self):
        ply_file = "reconstruction.ply"
        if os.path.exists(ply_file):
            viewer, ok = QInputDialog.getText(self, "Choisir le visualiseur", "Nom du programme (ex: meshlab, cloudcompare):")
            if ok and viewer:
                try:
                    subprocess.Popen([viewer, ply_file])
                    self.log_message(f"🌐 Ouverture de {ply_file} avec {viewer}...")
                except Exception as e:
                    self.log_message(f"❌ Erreur d'ouverture de {viewer} : {str(e)}")
            else:
                self.log_message("❌ Visualiseur non spécifié.")
        else:
            self.log_message("❌ Fichier reconstruction.ply non trouvé.")

    def load_left_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image gauche", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                self.log_message("❌ Erreur lors du chargement de l'image gauche.")
                return
            if self.calibration_loaded:
                img = cv2.undistort(img, self.K, self.dist)
                self.log_message("📷 Image gauche corrigée (distorsion supprimée).")
            self.left_image = img
            self.display_image(self.left_image, self.left_image_label)
            self.log_message("✅ Image gauche chargée.")

    def load_right_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image droite", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                self.log_message("❌ Erreur lors du chargement de l'image droite.")
                return
            if self.calibration_loaded:
                img = cv2.undistort(img, self.K, self.dist)
                self.log_message("📷 Image droite corrigée (distorsion supprimée).")
            self.right_image = img
            self.display_image(self.right_image, self.right_image_label)
            self.log_message("✅ Image droite chargée.")

    def capture_images(self):
        if hasattr(self, 'left_image') and hasattr(self, 'right_image'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path1 = os.path.join(self.capture_folder, f"left_{timestamp}.jpg")
            path2 = os.path.join(self.capture_folder, f"right_{timestamp}.jpg")
            cv2.imwrite(path1, self.left_image)
            cv2.imwrite(path2, self.right_image)
            self.log_message(f"📸 Capturé : {path1} et {path2}")
        else:
            self.log_message("❌ Images non disponibles à capturer.")

    def load_chessboard_folder_for_calibration(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner dossier d'échiquiers", os.getcwd())
        if not folder:
            self.log_message("❌ Aucun dossier sélectionné.")
            return
        self.log_message(f"📁 Dossier sélectionné : {folder}")
        try:
            result = calibrate_camera(folder, display=True)
            self.calibration_data = result
            self.K = result['K']
            self.dist = result['dist']
            self.calibration_loaded = True
            self.log_message("✅ Calibration réussie.")
            self.log_message(f"📷 Matrice K :\n{result['K']}")
            self.log_message(f"🎯 Erreur reprojection : {result['reprojection_error']:.4f}")
            save_calibration("calibration_result.npz", result)
        except Exception as e:
            self.log_message(f"❌ Erreur : {str(e)}")

    def calibrate_camera_from_folder(self):
        QMessageBox.information(self, "Instructions de Calibration", (
            "⚠️ Paramètres de calibration non trouvés.\n\n"
            "👉 Vous devez absolument effectuer une calibration avant de poursuivre.\n\n"
            "1️⃣ Si vous avez déjà un dossier contenant 6 images nettes d'un échiquier,\n"
            "   utilisez le bouton '📂 Charger dossier échiquiers'.\n\n"
            "2️⃣ Sinon, capturez 6 paires d'images avec '📸 Capturer'.\n"
            "   Les images sont enregistrées dans /home/roboticslab/Computer_Vision_JYV/captures/.\n"
            "   Copiez manuellement les images gauches ou droites dans un dossier dédié,\n"
            "   puis chargez ce dossier comme dans la méthode 1.\n\n"
            "📌 Une fois la calibration effectuée, vous pourrez charger vos images gauche/droite\n"
            "   et lancer le traitement 3D complet avec '⚙️ Traitement'.\n\n"
            "💡 Vous pouvez aussi utiliser vos propres paramètres de calibration en remplaçant\n"
            "   le fichier 'calibration_result.npz'."
        ))


    def process_stereo_vision(self):
        if not self.calibration_loaded:
            self.log_message("❌ Traitement annulé : calibration non effectuée.")
            return
        if not hasattr(self, 'left_image') or not hasattr(self, 'right_image'):
            self.log_message("❌ Traitement annulé : images non chargées.")
            return
        self.log_message("⚙️ Traitement en cours...")
        try:
            run_processing(self.left_image, self.right_image, self.K, self.dist, log_callback=self.log_message)
            self.log_message("✅ Traitement terminé avec succès.")
        except Exception as e:
            self.log_message(f"❌ Erreur pendant le traitement : {str(e)}")

    def display_image(self, image, label):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def log_message(self, msg):
        self.log_console.append(msg)

    def show_fullscreen_image(self, event):
        label = self.sender()
        if label == self.left_image_label and hasattr(self, 'left_image'):
            viewer = ImageViewer(self.left_image)
            viewer.show()
        elif label == self.right_image_label and hasattr(self, 'right_image'):
            viewer = ImageViewer(self.right_image)
            viewer.show()

    def closeEvent(self, event):
        if hasattr(self, 'cap1'): self.cap1.release()
        if hasattr(self, 'cap2'): self.cap2.release()
        if hasattr(self, 'timer'): self.timer.stop()
        super().closeEvent(event)