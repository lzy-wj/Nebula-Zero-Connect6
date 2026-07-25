import datetime
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QFormLayout, QLineEdit, 
                             QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt

class GameResultDialog(QDialog):
    """
    游戏结束时弹出的对话框，用于输入比赛信息并保存棋谱。
    """
    def __init__(self, winner, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保存棋谱 - 比赛信息")
        self.setMinimumWidth(450)
        self.setMinimumHeight(350)
        
        self.winner = winner  # 1=黑胜, -1=白胜, 0=平局, 2=流局（对局中保存）
        
        # 设置对话框样式
        self.setStyleSheet("""
            QDialog {
                background-color: #252526;
                color: #ffffff;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Microsoft YaHei UI', sans-serif;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                background-color: #333333;
                font-size: 14px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border-color: #007acc;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
            QPushButton {
                padding: 8px 20px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton[text="Save"], QPushButton[text="保存"] {
                background-color: #007acc;
                color: white;
                border: none;
            }
            QPushButton[text="Save"]:hover, QPushButton[text="保存"]:hover {
                background-color: #1f8ad2;
            }
            QPushButton[text="Cancel"], QPushButton[text="取消"] {
                background-color: #3e3e42;
                color: #cccccc;
                border: 1px solid #555;
            }
            QPushButton[text="Cancel"]:hover, QPushButton[text="取消"]:hover {
                background-color: #4e4e52;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 结果标题 (Big, White)
        if winner == 1:
            header_text = "🎉 黑方胜!"
            header_color = "#ffffff"
        elif winner == -1:
            header_text = "🎉 白方胜!"
            header_color = "#ffffff"
        elif winner == 2:
            header_text = "⏸️ 比赛暂停/流局"
            header_color = "#ffd700"
        else:
            header_text = "🤝 平局!"
            header_color = "#cccccc"

        title_label = QLabel(header_text)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {header_color}; margin-bottom: 5px;")
        layout.addWidget(title_label)
        
        # 副标题
        sub_label = QLabel("保存棋谱记录 (Game Record)")
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet("color: #888888; font-size: 13px; margin-bottom: 15px;")
        layout.addWidget(sub_label)
        
        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        self.edit_black_team = QLineEdit()
        self.edit_black_team.setPlaceholderText("输入先手参赛队名称")
        form_layout.addRow("先手 (黑方):", self.edit_black_team)
        
        self.edit_white_team = QLineEdit()
        self.edit_white_team.setPlaceholderText("输入后手参赛队名称")
        form_layout.addRow("后手 (白方):", self.edit_white_team)
        
        self.edit_location = QLineEdit()
        self.edit_location.setPlaceholderText("输入比赛地点")
        form_layout.addRow("比赛地点:", self.edit_location)
        
        self.edit_event = QLineEdit()
        self.edit_event.setPlaceholderText("输入赛事名称")
        form_layout.addRow("赛事名称:", self.edit_event)
        
        # 显示结果（只读）
        if winner == 1:
            result_text = "🏆 先手胜 (黑方获胜)"
            result_color = "#28a745"
        elif winner == -1:
            result_text = "🏆 后手胜 (白方获胜)"
            result_color = "#17a2b8"
        elif winner == 2:
            result_text = "⏸️ 流局 (对局中保存)"
            result_color = "#fd7e14"
        else:
            result_text = "🤝 平局"
            result_color = "#6c757d"
        self.lbl_result = QLabel(result_text)
        self.lbl_result.setStyleSheet(f"font-weight: bold; color: {result_color}; font-size: 15px;")
        form_layout.addRow("比赛结果:", self.lbl_result)
        
        # 显示时间（自动获取）
        self.game_time = datetime.datetime.now()
        time_text = self.game_time.strftime("%Y年%m月%d日 %H:%M")
        self.lbl_time = QLabel(f"🕐 {time_text}")
        self.lbl_time.setStyleSheet("color: #666; font-size: 13px;")
        form_layout.addRow("比赛时间:", self.lbl_time)
        
        layout.addLayout(form_layout)
        
        # 分隔线
        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #ddd;")
        layout.addWidget(line)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5c636a;
            }
        """)
        
        self.btn_save = QPushButton("保存棋谱")
        self.btn_save.setFixedWidth(120)
        self.btn_save.clicked.connect(self.accept)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        
        # 设置焦点到第一个输入框
        self.edit_black_team.setFocus()
        
    def get_game_info(self):
        """返回用户输入的比赛信息"""
        return {
            'black_team': self.edit_black_team.text().strip() or "先手队",
            'white_team': self.edit_white_team.text().strip() or "后手队",
            'location': self.edit_location.text().strip() or "未知",
            'event': self.edit_event.text().strip() or "友谊赛",
            'winner': self.winner,
            'time': self.game_time
        }
