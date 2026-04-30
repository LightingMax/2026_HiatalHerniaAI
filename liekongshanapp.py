import multiprocessing
import os
import sys
import time
import warnings

if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    # Help Windows loader find torch native DLLs in frozen app runtime dir.
    torch_lib_dir = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.isdir(torch_lib_dir):
        os.add_dll_directory(torch_lib_dir)

import pandas as pd
import timm
import torch
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplashScreen,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torchvision import transforms


def resource_path(relative_path):
    """获取资源绝对路径，兼容源码运行和 PyInstaller 打包运行。"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class HerniaAIEngine:
    def __init__(self, model_path, num_classes=4):
        # 按需求固定为 CPU 推理，确保在 Windows 无显卡环境可运行
        self.device = torch.device("cpu")
        self.img_size = 384
        self.class_names = [f"级别 {idx}" for idx in range(1, num_classes + 1)]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.model = timm.create_model(
            "timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
            pretrained=False,
            num_classes=num_classes,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

        state_dict = self._safe_load_state_dict(model_path)

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _safe_load_state_dict(self, model_path):
        """
        Try secure loading first (weights_only=True). If checkpoint format is not
        compatible and the file is trusted, fallback to weights_only=False.
        """
        # Detect Git LFS pointer file early (common cause of "invalid load key, 'v'")
        with open(model_path, "rb") as f:
            head = f.read(200)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                "检测到模型文件是 Git LFS 指针而非真实权重文件。"
                "请在构建环境执行 git lfs pull，或确保将真实 .pth 文件打包进 EXE。"
            )

        checkpoint = None
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Backward compatibility with very old torch versions
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as safe_err:
            warnings.warn(
                "Safe checkpoint load failed (weights_only=True). "
                "Falling back to weights_only=False for trusted local model file only. "
                f"Original error: {safe_err}"
            )
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Common checkpoint structures
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            raise RuntimeError("模型文件格式无法解析为 state_dict，请检查权重文件来源与格式。")

        return checkpoint

    def predict(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor_img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor_img)
                prob = torch.softmax(output, dim=1)
                pred_idx = int(prob.argmax(dim=1).item())
                confidence = float(prob[0, pred_idx].item())
            return self.class_names[pred_idx], confidence
        except Exception as e:
            return f"错误: {e}", None


class InferenceWorker(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(int, str)
    finished = pyqtSignal(int)

    def __init__(self, engine, tasks):
        super().__init__()
        self.engine = engine
        self.tasks = tasks

    def run(self):
        total = len(self.tasks)
        done = 0
        for row_idx, img_path in self.tasks:
            label, confidence = self.engine.predict(img_path)
            if confidence is None:
                display = label
            else:
                display = f"{label} (置信度 {confidence * 100:.1f}%)"
            self.result_ready.emit(row_idx, display)
            done += 1
            self.progress_updated.emit(int(done / total * 100))
        self.finished.emit(done)


class ModernSplashScreen(QSplashScreen):
    def __init__(self):
        pixmap = QPixmap(600, 350)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 600, 350)
        gradient.setColorAt(0.0, QColor(20, 30, 48))
        gradient.setColorAt(1.0, QColor(36, 59, 85))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 600, 350, 20, 20)

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        painter.drawText(
            pixmap.rect(),
            Qt.AlignCenter,
            "食道裂孔疝 AI 辅助诊断系统\nHiatal Hernia AI System",
        )
        painter.end()

        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

    def show_message(self, message):
        self.showMessage(message, Qt.AlignBottom | Qt.AlignCenter, QColor(200, 200, 200))


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.records = []
        self.current_row = -1
        self.worker = None
        self.current_batch_size = 0
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("食道裂孔疝 AI 辅助诊断工作站 - CPU 版")
        self.resize(1180, 720)

        icon_path = resource_path("app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setStyleSheet(
            """
            QWidget { font-family: 'Microsoft YaHei'; font-size: 13px; }
            QMainWindow { background: #f6f8fb; }
            QPushButton {
                background: #1f5fae;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton:hover { background: #174b8a; }
            QPushButton:disabled { background: #9cb6d5; }
            QTableWidget {
                background: #ffffff;
                gridline-color: #e3e8ef;
                border: 1px solid #dbe4f0;
                border-radius: 8px;
            }
            QHeaderView::section {
                background: #edf3fb;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #dbe4f0;
                font-weight: 600;
            }
            QLabel#preview {
                background-color: #ffffff;
                border: 1px solid #dbe4f0;
                border-radius: 8px;
                color: #6a7687;
            }
            QProgressBar {
                border: 1px solid #dbe4f0;
                border-radius: 6px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background-color: #2f80ed;
            }
            """
        )

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        left_panel = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_open_file = QPushButton("打开单张图片")
        self.btn_open_folder = QPushButton("导入文件夹")
        self.btn_export = QPushButton("导出诊断报告")

        self.btn_open_file.clicked.connect(self.load_single_image)
        self.btn_open_folder.clicked.connect(self.load_folder)
        self.btn_export.clicked.connect(self.export_results)

        btn_layout.addWidget(self.btn_open_file)
        btn_layout.addWidget(self.btn_open_folder)
        btn_layout.addWidget(self.btn_export)
        left_panel.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["图片名", "AI 诊断", "医生复核"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self.on_table_click)
        left_panel.addWidget(self.table)

        right_panel = QVBoxLayout()
        self.lbl_image = QLabel("请在左侧导入并选择图片")
        self.lbl_image.setObjectName("preview")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(520, 520)
        right_panel.addWidget(self.lbl_image)

        doctor_layout = QHBoxLayout()
        doctor_layout.addWidget(QLabel("医生人工复核诊断:"))
        self.combo_doctor = QComboBox()
        self.combo_doctor.addItems(["未诊断", "级别 1", "级别 2", "级别 3", "级别 4"])
        self.combo_doctor.currentTextChanged.connect(self.on_doctor_diagnosis_changed)
        self.combo_doctor.setEnabled(False)
        doctor_layout.addWidget(self.combo_doctor)
        right_panel.addLayout(doctor_layout)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.lbl_status = QLabel("就绪 | 推理设备: CPU")
        status_bar.addWidget(self.lbl_status)

    def set_busy(self, busy):
        self.btn_open_file.setEnabled(not busy)
        self.btn_open_folder.setEnabled(not busy)

    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.add_tasks([file_path])

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if folder_path:
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            paths = [
                os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files
                if os.path.splitext(file)[1].lower() in valid_exts
            ]
            if paths:
                self.add_tasks(paths)
            else:
                QMessageBox.warning(self, "提示", "该文件夹下未找到受支持的图片格式。")

    def add_tasks(self, file_paths):
        start_row = self.table.rowCount()
        tasks_for_worker = []
        self.current_batch_size = len(file_paths)
        for i, path in enumerate(file_paths):
            row = start_row + i
            filename = os.path.basename(path)
            self.records.append(
                {"path": path, "filename": filename, "ai": "推理中...", "doctor": "未诊断"}
            )
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(filename))
            self.table.setItem(row, 1, QTableWidgetItem("推理中..."))
            self.table.setItem(row, 2, QTableWidgetItem("未诊断"))
            tasks_for_worker.append((row, path))

        self.set_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"正在推理: 0/{self.current_batch_size}")

        self.worker = InferenceWorker(self.engine, tasks_for_worker)
        self.worker.result_ready.connect(self.update_ai_result)
        self.worker.progress_updated.connect(self.on_progress)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()

    def on_progress(self, value):
        self.progress_bar.setValue(value)
        done = int((value / 100) * self.current_batch_size)
        self.lbl_status.setText(f"正在推理: {done}/{self.current_batch_size}")

    def on_inference_finished(self, done):
        self.progress_bar.setVisible(False)
        self.set_busy(False)
        self.lbl_status.setText(f"推理完成，本轮处理 {done} 张，累计 {len(self.records)} 张")

    def update_ai_result(self, row_idx, result):
        self.records[row_idx]["ai"] = result
        item = QTableWidgetItem(result)
        if "级别 4" in result:
            item.setForeground(QColor("#cc2f2f"))
        elif "级别 3" in result:
            item.setForeground(QColor("#e67e22"))
        elif "级别 2" in result:
            item.setForeground(QColor("#1f5fae"))
        elif "级别 1" in result:
            item.setForeground(QColor("#2e7d32"))
        self.table.setItem(row_idx, 1, item)

    def on_table_click(self, row, _column):
        self.current_row = row
        record = self.records[row]
        pixmap = QPixmap(record["path"])
        scaled_pixmap = pixmap.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(scaled_pixmap)

        self.combo_doctor.blockSignals(True)
        self.combo_doctor.setEnabled(True)
        self.combo_doctor.setCurrentText(record["doctor"])
        self.combo_doctor.blockSignals(False)

    def on_doctor_diagnosis_changed(self, text):
        if self.current_row >= 0:
            self.records[self.current_row]["doctor"] = text
            self.table.setItem(self.current_row, 2, QTableWidgetItem(text))

    def export_results(self):
        if not self.records:
            QMessageBox.information(self, "提示", "目前没有可导出的数据。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "食道裂孔疝诊断报告.xlsx", "Excel Files (*.xlsx)"
        )
        if save_path:
            df = pd.DataFrame(self.records)[["filename", "path", "ai", "doctor"]]
            df.columns = ["图片名", "本地路径", "AI自动诊断", "医生人工复核"]
            try:
                df.to_excel(save_path, index=False)
                QMessageBox.information(self, "成功", f"报告已成功导出至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")


def main():
    multiprocessing.freeze_support()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon_path = resource_path("app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash = ModernSplashScreen()
    splash.show()
    app.processEvents()

    model_weight_path = resource_path("best_model_liekongshan.pth")

    splash.show_message("正在加载 CPU 推理环境...")
    time.sleep(0.4)

    try:
        splash.show_message("正在载入模型参数，这可能需要几十秒...")
        app.processEvents()
        engine = HerniaAIEngine(model_path=model_weight_path, num_classes=4)

        splash.show_message("模型加载完毕，正在启动界面...")
        app.processEvents()
        time.sleep(0.3)

        main_window = MainWindow(engine)
        main_window.show()
        splash.finish(main_window)
        return app.exec_()
    except Exception as e:
        QMessageBox.critical(None, "致命错误", f"模型加载失败，请检查路径或环境:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
