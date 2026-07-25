import sys
import os
import time
import numpy as np # Import numpy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QSpinBox, QFileDialog, QMessageBox, 
                             QGroupBox, QCheckBox, QTextEdit, QComboBox, QInputDialog,
                             QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QFrame,
                             QRadioButton, QButtonGroup, QScrollArea)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent, QPainterPath, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
import datetime

# Import refactored components
try:
    from ui.game_board import Connect6Board, format_move_coord, BOARD_SIZE, MARGIN
    from ui.dialogs import GameResultDialog
except ImportError:
    # Fail-safe for different running contexts
    try:
        from game_board import Connect6Board, format_move_coord, BOARD_SIZE, MARGIN
        from dialogs import GameResultDialog
    except ImportError:
        # Assuming local structure
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from game_board import Connect6Board, format_move_coord, BOARD_SIZE, MARGIN
        from dialogs import GameResultDialog

from sgf_handler import C6SGFHandler
from ai_interface import AIWorker

WINDOW_WIDTH = 1570
WINDOW_HEIGHT = 1150

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nebula-Zero——Connect6 AI")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "photos", "Nebulazero.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # State
        self.moves = [] # List of (r, c, player)
        self.move_evaluations = {}  # Dict mapping move_idx -> MARK value (-2 to 2)
        self.last_ai_win_rate = None  # Track AI's win rate after each move
        self.history_states = [] # For undo
        self.current_player = 1 # Black
        self.human_role = 1 # 1: Black, -1: White (MUST be initialized here!)
        self.game_active = False
        self.time_black = 600 # 10 mins
        self.time_white = 600
        self.turn_moves_left = 1 # First move is 1 stone
        self.is_ai_thinking = False # Initialize state
        self.game_id = 0 # Game Session ID to prevent race conditions
        
        # Team Mode State
        self.team_rotation_mode = False  # True = 团队轮换模式
        self.team_is_black = True  # True = 我方是黑方
        self.white_start_human = True  # True = 白方人类先手
        self.operator_time_limit = 30  # Seconds
        self.human_turn_timer = 30  # Current countdown for human turn
        self.reference_mode = False  # True = AI analyzing for reference only, don't make moves

        
        # AI
        self.ai_worker = None
        self.engine_path = "engine/current_model.engine" # Relative path
        
        # Initialize AI Worker (Persistent Thread)
        self.ai_worker = AIWorker(self.engine_path)
        self.ai_worker.update_stats.connect(self.on_ai_stats)
        self.ai_worker.decision_made.connect(self.on_ai_decision) # Now accepts (r, c, game_id)
        self.ai_worker.start()
        
        self.init_ui()
        self.apply_styles()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)
        
        # AI 触发定时器 (可控，用于替代 singleShot)
        self.ai_trigger_timer = QTimer()
        self.ai_trigger_timer.setSingleShot(True)
        self.ai_trigger_timer.timeout.connect(self.trigger_ai)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            }
            QLabel, QRadioButton, QCheckBox {
                color: #cccccc;
                font-size: 15px;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 8px;
                margin-top: 24px;
                font-weight: bold;
                font-size: 16px;
                color: #e0e0e0;
                background-color: #252526;
            }
            QGroupBox#info {
                border: 1px solid #3e3e42;
                border-radius: 8px;
                margin-top: 24px;
                font-weight: bold;
                font-size: 20px;
                color: #e0e0e0;
                background-color: #252526;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                left: 10px;
                color: #aeaeae;
                font-size: 16px;
            }
            
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #1f8ad2;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #858585;
                border: none;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                padding: 6px;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                background-color: #333333;
                color: #f0f0f0;
                font-size: 16px;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                selection-background-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                background: transparent;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #3e3e42;
                background-color: #252526;
                color: #f0f0f0;
                font-size: 16px;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                selection-background-color: #007acc;
                outline: 0px;
            }
            QComboBox QAbstractItemView::item {
                background-color: #252526;
                color: #f0f0f0;
                padding: 6px;
                min-height: 24px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3e3e42;
                color: #ffffff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #007acc;
                color: #ffffff;
            }
            QComboBox QListView {
                background-color: #252526;
                color: #f0f0f0;
                border: 1px solid #3e3e42;
            }
            QCheckBox {
                color: #cccccc;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #555;
                background-color: #333;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QMessageBox {
                background-color: #252526;
                color: #e0e0e0;
            }
            QMessageBox QLabel {
                color: #e0e0e0;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #424242;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Specific overrides
        self.lbl_black_time.setStyleSheet("color: #ff6b6b; font-size: 18px; font-weight: bold;")
        self.lbl_white_time.setStyleSheet("color: #4ecdc4; font-size: 18px; font-weight: bold;")
        self.lbl_turn.setStyleSheet("color: #ffe66d; font-size: 16px; font-weight: bold;")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Side: Board + Top Notification
        left_panel = QVBoxLayout()
        
        # Top Notification Bar (Human Turn Reminder)
        self.notification_bar = QLabel("等待对手...")
        self.notification_bar.setAlignment(Qt.AlignCenter)
        self.notification_bar.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #888888;
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.notification_bar.setFixedHeight(50)
        left_panel.addWidget(self.notification_bar)
        
        # Board (Resizable)
        self.board_widget = Connect6Board()
        self.board_widget.move_signal.connect(self.handle_player_move)
        left_panel.addWidget(self.board_widget, 1)  # Board takes expanding space
        
        main_layout.addLayout(left_panel, 1)
        
        # Right: Controls (Scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(500)  # Increased width
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Disable horizontal scrollbar
        # Remove borders and background
        right_scroll.setStyleSheet("""
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical { width: 8px; background: #2d2d2d; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; }
        """)
        
        right_panel = QFrame()
        right_panel.setObjectName("RightPanel")
        right_panel.setStyleSheet("#RightPanel { background-color: transparent; border: none; }") # Ensure no border on frame
        
        control_panel = QVBoxLayout(right_panel)
        control_panel.setContentsMargins(10, 0, 20, 0) # Right margin increased
        control_panel.setSpacing(10) # Reduce spacing
        
        right_scroll.setWidget(right_panel)
        main_layout.addWidget(right_scroll)
        
        # 1. Info Group
        info_group = QGroupBox("对局信息")
        info_group.setObjectName("info")
        info_layout = QVBoxLayout()
        self.lbl_black_time = QLabel("黑方: 10:00")
        self.lbl_white_time = QLabel("白方: 10:00")
        self.lbl_turn = QLabel("当前: 黑方 (1子)")
        
        # Use HTML for rich text color
        self.lbl_black_time.setTextFormat(Qt.RichText)
        self.lbl_white_time.setTextFormat(Qt.RichText)
        self.lbl_turn.setTextFormat(Qt.RichText)
        
        info_layout.addWidget(self.lbl_black_time)
        info_layout.addWidget(self.lbl_white_time)
        info_layout.addWidget(self.lbl_turn)
        
        self.lbl_operator = QLabel("操作者: --")
        self.lbl_operator.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 18px;")
        info_layout.addWidget(self.lbl_operator)
        info_group.setLayout(info_layout)
        control_panel.addWidget(info_group)
        
        # 2. AI Stats
        ai_group = QGroupBox("AI 状态")
        ai_layout = QVBoxLayout()
        self.lbl_ai_status = QLabel("💤 空闲")
        self.lbl_winrate = QLabel("📊 胜率: --")
        self.lbl_sims = QLabel("🔄 模拟: 0")
        self.lbl_debug_info = QLabel("🔧 Debug: Temp=--") 
        
        ai_layout.addWidget(self.lbl_ai_status)
        ai_layout.addWidget(self.lbl_winrate)
        ai_layout.addWidget(self.lbl_sims)
        ai_layout.addWidget(self.lbl_debug_info)
        ai_group.setLayout(ai_layout)
        control_panel.addWidget(ai_group)
        
        # 3. Game Mode Settings
        settings_group = QGroupBox("比赛设置")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)
        
        # 1. Game Mode
        settings_layout.addWidget(QLabel("游戏模式 (Game Mode):"))
        self.rb_normal_mode = QRadioButton("普通模式 (Normal)")
        self.rb_team_mode = QRadioButton("团队轮换 (Team Rotation)")
        self.rb_normal_mode.setChecked(True)
        self.rb_normal_mode.toggled.connect(self.on_game_mode_changed)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.rb_normal_mode)
        mode_layout.addWidget(self.rb_team_mode)
        settings_layout.addLayout(mode_layout)
        
        # 2. Side Selection (AI Centric)
        settings_layout.addWidget(QLabel("AI 阵营 (AI Side):"))
        self.combo_side = QComboBox()
        self.combo_side.addItem("AI 执黑 (先手)", 1)
        self.combo_side.addItem("AI 执白 (后手)", -1)
        self.combo_side.currentIndexChanged.connect(self.on_settings_changed)
        settings_layout.addWidget(self.combo_side)
        
        # 3. White Start Option (Conditional)
        self.lbl_white_start = QLabel("白方首手 (White 1st):")
        settings_layout.addWidget(self.lbl_white_start)
        
        self.combo_white_start = QComboBox()
        self.combo_white_start.addItem("人类", True)
        self.combo_white_start.addItem("AI", False)
        settings_layout.addWidget(self.combo_white_start)
        
        # Dummy Legacy Variables to prevent AttributeErrors if referenced elsewhere
        self.combo_player = self.combo_side # Map legacy to new
        self.rb_team_black = None 
        self.rb_team_mode = self.rb_team_mode # Keep reference


        settings_layout.addWidget(QLabel("模拟次数 (Simulations):"))
        self.spin_sims = QSpinBox()
        self.spin_sims.setRange(0, 50000)
        self.spin_sims.setValue(12000)
        self.spin_sims.valueChanged.connect(self.update_ai_params)
        settings_layout.addWidget(self.spin_sims)
        
        # Dynamic Thinking Mode
        self.chk_dynamic = QCheckBox("动态思考 (Dynamic Think)")
        self.chk_dynamic.setChecked(True)
        self.chk_dynamic.stateChanged.connect(self.update_ai_params)
        settings_layout.addWidget(self.chk_dynamic)
        
        # Deep Thinking Mode
        self.chk_deep = QCheckBox("深度思考 (Deep Think)")
        self.chk_deep.setChecked(True)
        self.chk_deep.stateChanged.connect(self.update_ai_params)
        settings_layout.addWidget(self.chk_deep)

        self.chk_ponder = QCheckBox("后台思考 (Background Pondering)")
        self.chk_ponder.setChecked(True)
        self.chk_ponder.stateChanged.connect(self.update_ai_params)
        settings_layout.addWidget(self.chk_ponder)
        
        # Opening Book Toggle (only for Normal mode)
        self.chk_opening_book = QCheckBox("开局库 (Opening Book)")
        self.chk_opening_book.setChecked(False)  # 默认关闭
        self.chk_opening_book.setEnabled(True)   # 普通模式默认可用
        settings_layout.addWidget(self.chk_opening_book)
        
        # Initialize UI State (Disable white_start by default)
        self.lbl_white_start.setEnabled(False)
        self.combo_white_start.setEnabled(False)
        
        settings_group.setLayout(settings_layout)
        control_panel.addWidget(settings_group)
        
        # 4. Buttons
        btn_layout = QVBoxLayout()
        
        self.btn_new = QPushButton("新对局 (New Game)")
        self.btn_new.clicked.connect(self.start_game)
        
        self.btn_pause = QPushButton("暂停 (Pause)")
        self.btn_pause.clicked.connect(self.toggle_pause)
        
        self.btn_undo = QPushButton("悔棋 (Undo)")
        self.btn_undo.clicked.connect(self.undo_move)
        self.btn_undo.setEnabled(True)  # 确保初始状态为启用
        
        self.btn_add_time = QPushButton("加时 (Add)")
        self.btn_add_time.clicked.connect(self.add_time)
        
        self.btn_save = QPushButton("保存棋谱 (Save)")
        self.btn_save.clicked.connect(self.save_game)
        
        self.btn_load = QPushButton("载入棋谱 (Load)")
        self.btn_load.clicked.connect(self.load_game)
        
        # Removed Force AI button as requested
        # self.btn_ai_move = QPushButton("强制 AI 落子 (Force AI)")
        # self.btn_ai_move.clicked.connect(self.trigger_ai)

        self.btn_show_policy = QPushButton("显示策略热力图 (Show Policy)")
        self.btn_show_policy.setCheckable(True)
        self.btn_show_policy.clicked.connect(self.toggle_policy_view)

        # Debug Pause Feature
        self.chk_debug_pause = QCheckBox("调试暂停 (Pause Before Move)")
        self.chk_debug_pause.setChecked(False)
        
        self.btn_confirm_move = QPushButton("确认落子 (Confirm Move)")
        self.btn_confirm_move.setEnabled(False)
        self.btn_confirm_move.clicked.connect(self.confirm_ai_move)
        self.pending_ai_move = None # Store (r, c)

        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_undo)
        btn_layout.addWidget(self.btn_add_time)
        # btn_layout.addWidget(self.btn_ai_move) # Removed
        btn_layout.addWidget(self.btn_show_policy)
        btn_layout.addWidget(self.chk_debug_pause) # New Checkbox
        btn_layout.addWidget(self.btn_confirm_move) # New Button
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_load)
        
        control_panel.addLayout(btn_layout)
        control_panel.addStretch()

        # Initial param sync
        self.update_ai_params()
        
    def on_game_mode_changed(self, checked):
        """Toggle Game Mode and Update UI"""
        self.on_settings_changed()
        
    def on_settings_changed(self):
        """Unified Handler for Settings Changes"""
        is_team_mode = self.rb_team_mode.isChecked()
        side_idx = self.combo_side.currentIndex() # 0=Black, 1=White
        is_my_side_white = (side_idx == 1)
        
        # White Start Option Logic
        # Enable only if Team Mode AND My Side is White
        enable_white_start = is_team_mode and is_my_side_white
        
        self.lbl_white_start.setEnabled(enable_white_start)
        self.combo_white_start.setEnabled(enable_white_start)
        
        # Opening Book Logic
        # 团队模式下禁用开局库，自动取消选中
        if is_team_mode:
            self.chk_opening_book.setChecked(False)
            self.chk_opening_book.setEnabled(False)
        else:
            self.chk_opening_book.setEnabled(True)
        
        # If disabled, maybe reset to clear confusion? Or keep as is.
        # Keeping as is is fine.

    def get_current_operator(self):
        """
        Calculate who should operate the current turn.
        Returns: 'Human' or 'AI'
        """
        if not self.game_active: return 'Human'
        
        # Read current settings directly
        is_team_mode = self.team_rotation_mode # State var updated in start_game
        # Note: We should use the state variables set in start_game, NOT UI widgets directly during game loop
        # ensuring consistency even if user toggles UI during game
        
        # Black's Turn
        if self.current_player == 1:
            if is_team_mode and self.team_is_black:
                # My Team is Black (Rotation)
                stones_on_board_black = sum(1 for r,c,p in self.moves if p == 1)
                if stones_on_board_black == 0: turn_idx = 0
                elif stones_on_board_black == 1: turn_idx = 1
                else: turn_idx = 2 + (stones_on_board_black - 3) // 2
                return 'Human' if (turn_idx % 2 == 0) else 'AI'
                
            elif is_team_mode and not self.team_is_black:
                # My Team is White, Opponent (Black) is Human
                return 'Human'
            else:
                # Normal Mode: Black Turn
                # If human_role is Black (1), then Human. Else AI.
                return 'Human' if self.human_role == 1 else 'AI'
                
        # White's Turn
        else:
            if is_team_mode and not self.team_is_black:
                # My Team is White (Rotation)
                stones_on_board_white = sum(1 for r,c,p in self.moves if p == -1)
                turn_idx = stones_on_board_white // 2
                
                start_is_human = self.white_start_human # State var
                current_is_human = (turn_idx % 2 == 0) if start_is_human else (turn_idx % 2 != 0)
                return 'Human' if current_is_human else 'AI'
                
            elif is_team_mode and self.team_is_black:
                # My Team is Black, Opponent (White) is Human
                return 'Human'
            else:
                # Normal Mode: White Turn
                return 'Human' if self.human_role == -1 else 'AI'
        


    def update_operator_label(self):
        op = self.get_current_operator()
        self.lbl_operator.setText(f"操作者: {op}")
        if op == 'Human':
            self.lbl_operator.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 18px;")
        else:
            self.lbl_operator.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 18px;")
        
        # Update notification bar
        self.update_notification_bar()
    
    def update_notification_bar(self):
        """更新顶部通知栏"""
        if not self.game_active:
            self.notification_bar.setText("等待开始...")
            self.notification_bar.setStyleSheet("""
                QLabel { background-color: #2d2d2d; color: #888888; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
            """)
            return
            
        op = self.get_current_operator()
        
        # 判断是否是我方回合（团队模式）
        # 我方是黑方(True) 且 当前是黑方(1) -> True
        # 我方是白方(False) 且 当前是白方(-1) -> True
        my_color = 1 if self.team_is_black else -1
        is_my_turn_team = (self.current_player == my_color)
        
        # 团队轮换模式逻辑
        if self.team_rotation_mode:
            if is_my_turn_team:
                if op == 'Human':
                    # 我方人类回合 - 绿色 + 倒计时
                    time_left = self.human_turn_timer
                    if time_left <= 10:
                        self.notification_bar.setText(f"你的回合！剩余 {time_left} 秒")
                        self.notification_bar.setStyleSheet("""
                            QLabel { background-color: #ff4444; color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
                        """)
                    else:
                        self.notification_bar.setText(f"你的回合！剩余 {time_left} 秒")
                        self.notification_bar.setStyleSheet("""
                            QLabel { background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
                        """)
                else: 
                    # 我方AI回合 - 蓝色
                    self.notification_bar.setText("AI 思考中...")
                    self.notification_bar.setStyleSheet("""
                        QLabel { background-color: #2196F3; color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
                    """)
            else:
                # 对方回合（总是人类） - 黑色/白色背景
                turn_str = "黑方" if self.current_player == 1 else "白方"
                # 黑底白字 或者 白底黑字
                bg_color = "#000000" if self.current_player == 1 else "#f0f0f0"
                text_color = "#ffffff" if self.current_player == 1 else "#000000"
                border_color = "#333" if self.current_player == 1 else "#ccc"
                
                self.notification_bar.setText(f"对方回合 ({turn_str})")
                self.notification_bar.setStyleSheet(f"""
                    QLabel {{ background-color: {bg_color}; color: {text_color}; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; border: 1px solid {border_color}; }}
                """)
        else:
            # 普通模式
            if op == 'AI':
                self.notification_bar.setText("AI 思考中...")
                self.notification_bar.setStyleSheet("""
                    QLabel { background-color: #2196F3; color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
                """)
            else:
                # 人类回合（我方）
                self.notification_bar.setText(f"轮到你了")
                self.notification_bar.setStyleSheet("""
                    QLabel { background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
                """)
            
    def check_ai_turn(self):
        self.update_operator_label()
        
        if not self.game_active or self.is_paused: return
        
        operator = self.get_current_operator()
        
        if operator == 'AI':
            self.reference_mode = False
            self.board_widget.interaction_enabled = False
            self.ai_trigger_timer.start(500)
        else:
            # Human Turn - Reset timer
            self.human_turn_timer = self.operator_time_limit
            self.board_widget.interaction_enabled = True
            
            # 团队模式：在我方人类回合启动参考分析（显示热力图）
            if self.team_rotation_mode:
                my_color = 1 if self.team_is_black else -1
                is_my_turn = (self.current_player == my_color)
                if is_my_turn:
                    # 启动参考模式分析 - AI 分析但不落子
                    self.reference_mode = True
                    self.start_reference_analysis()
    
    def start_reference_analysis(self):
        """启动参考分析：AI 分析当前局面，更新热力图，但不落子"""
        if not self.game_active: return
        
        # 设置AI颜色为当前玩家
        self.ai_worker.set_ai_color(self.current_player)
        
        # 准备历史
        hist = []
        for r, c, p in self.moves:
            hist.append(r * 19 + c)
        
        # 发起分析请求 - 结果会通过 on_ai_stats 更新热力图
        # on_ai_decision 会忽略参考模式的决策
        self.ai_worker.request_move(hist, self.current_player, self.game_id)
        self.lbl_ai_status.setText("状态: 参考分析中...")

    def on_player_side_change(self, index):
        """处理玩家选择黑白手的变化，对局中禁止切换"""
        new_role = self.combo_player.currentData()
        
        # 检查是否允许切换
        if not self.can_change_player_side():
            # 恢复到之前的选择
            old_index = 0 if self.human_role == 1 else 1
            self.combo_player.blockSignals(True)  # 阻止递归触发
            self.combo_player.setCurrentIndex(old_index)
            self.combo_player.blockSignals(False)
            
            # 使用美化的消息框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("无法切换")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("⚠️ 对局进行中不能切换黑白手！")
            msg_box.setInformativeText("请先结束当前对局或点击「新对局」按钮。")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #252526;
                    color: #e0e0e0;
                }
                QMessageBox QLabel {
                    color: #e0e0e0;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: none;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #1f8ad2;
                }
            """)
            msg_box.exec_()
            return
        
        # 允许切换，更新状态
        self.human_role = new_role
        print(f"🔄 [Settings] Player side changed to {'Black' if new_role == 1 else 'White'}")
    
    def can_change_player_side(self):
        """判断是否允许切换黑白手"""
        # 情况1: 棋盘为空（没有任何着法）
        if len(self.moves) == 0:
            return True
        
        # 情况2: 游戏已结束（game_active = False）
        if not self.game_active:
            return True
        
        # 情况3: 对局进行中，禁止切换
        return False
    
    def toggle_policy_view(self):
        self.board_widget.show_policy = self.btn_show_policy.isChecked()
        self.board_widget.update()

    def _clear_heatmap(self):
        """清空热力图数据"""
        self.board_widget.policy_data = []
        self.board_widget.update()
        
    def confirm_ai_move(self):
        # 如果 AI 正在思考，强制停止并落子
        if self.is_ai_thinking:
            self.ai_worker.finish_thinking()
            self.btn_confirm_move.setEnabled(False)
            self.lbl_ai_status.setText("状态: 正在落子...")
            return

        if self.pending_ai_move:
            r, c = self.pending_ai_move
            self.pending_ai_move = None
            self.btn_confirm_move.setEnabled(False)
            self.execute_ai_move(r, c)

    def update_ai_params(self):
        sims = self.spin_sims.value()
        
        # Hardcoded: Batch=32, Threads=8, Temp=0
        batch_size = 32
        threads = 8
        temp = 0.0
        
        # Read from checkboxes
        dynamic_think = self.chk_dynamic.isChecked()
        deep_think = self.chk_deep.isChecked()
        
        self.ai_worker.update_params(
            batch_size, 
            threads, 
            sims, 
            dynamic_think,
            temp
        )
        self.ai_worker.set_ponder(self.chk_ponder.isChecked())
        self.ai_worker.set_deep_thinking(deep_think)
        
        # Update UI if AI is currently thinking
        if self.is_ai_thinking:
            self.lbl_ai_status.setText("状态: AI 思考中...")
            self.btn_confirm_move.setText("确认落子 (Confirm Move)")
            # If not in debug pause, disable button (wait for AI to finish naturally)
            if not self.chk_debug_pause.isChecked():
                self.btn_confirm_move.setEnabled(False)

    def start_game(self):
        # 先停止可能在等待的 AI 触发定时器
        self.ai_trigger_timer.stop()
        self.game_id += 1 # Increment Game ID (New Session)
        print(f"🎬 [Game] Start New Game (ID: {self.game_id})")
        
        self.moves = []
        self.board_widget.board.fill(0)
        self.board_widget.policy_data = []  # 清空热力图数据
        self.board_widget.current_player = 1 # Reset board player
        self.board_widget.update()
        self.current_player = 1
        self.turn_moves_left = 1 # Black first 1
        self.time_black = 600
        self.time_white = 600
        self.game_active = True
        self.is_paused = False
        self.is_ai_thinking = False # Flag to prevent double triggering
        self.btn_pause.setText("暂停 (Pause)")
        self.board_widget.interaction_enabled = True
        
        # Get AI Side directly
        # combo_side data: 1 = AI is Black, -1 = AI is White
        ai_side = self.combo_side.currentData()
        
        # human_role is the opposite of AI's side
        self.human_role = -ai_side  # AI执黑(1) -> 人类执白(-1)
        
        # Game Mode Settings
        self.team_rotation_mode = self.rb_team_mode.isChecked()
        
        # In Team Mode:
        # If combo_side is 1 (AI Black), my team is Black (AI helps me).
        # If combo_side is -1 (AI White), my team is White.
        self.team_is_black = (ai_side == 1)
        
        # White Start Human Setting (only relevant for White Team in Team Mode)
        self.white_start_human = self.combo_white_start.currentData() 
        # Note: combo_white_start stores True (Human) or False (AI)
        
        # Calculate AI Color FIRST (before reset)
        if self.team_rotation_mode:
            # 团队轮换模式：AI 协助我方
            # AI是黑(1) -> 我方是黑
            # AI是白(-1) -> 我方是白
            ai_color = 1 if self.team_is_black else -1
        else:
            # 普通模式
            # AI color is directly from combo_side
            ai_color = ai_side  # 直接使用选择的颜色
        
        # Reset AI State WITH ai_color to avoid timing issues
        self.ai_worker.reset_game(ai_color)
        
        # 同步开局库设置给 AI（通过 checkbox 控制，团队模式下强制关闭）
        use_opening_book = self.chk_opening_book.isChecked() and not self.team_rotation_mode
        self.ai_worker.set_opening_book_enabled(use_opening_book)
        
        # 强制等待一下，确保 RESET 入队 (Fix Queue Race)
        import time
        time.sleep(0.05)
        
        self.update_turn_label()
        
        # Check if AI needs to move first
        self.check_ai_turn()

    def closeEvent(self, event):
        self.ai_worker.stop()
        event.accept()

    def toggle_pause(self):
        if not self.game_active: return
        
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.setText("继续 (Resume)")
            self.board_widget.interaction_enabled = False
            self.lbl_ai_status.setText("状态: 暂停中")
        else:
            self.btn_pause.setText("暂停 (Pause)")
            self.lbl_ai_status.setText("状态: 恢复")
            # 恢复时根据当前操作者决定行为
            operator = self.get_current_operator()
            if operator == 'Human':
                # 人类轮次：启用棋盘交互
                self.board_widget.interaction_enabled = True
            else:
                # AI 轮次：重新触发 AI 思考
                self.check_ai_turn()

    def add_time(self):
        # Allow adding time even if game is not active (e.g. timeout)
        dialog = QInputDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Remove ? button
        dialog.setWindowTitle("加时")
        dialog.setLabelText("输入加时秒数:")
        dialog.setIntRange(1, 3600)
        dialog.setIntValue(60)
        dialog.setStyleSheet("""
            QInputDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
            QSpinBox {
                background-color: #333333;
                color: #f0f0f0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1f8ad2;
            }
        """)
        
        if dialog.exec_() == QInputDialog.Accepted:
            seconds = dialog.intValue()
            if self.current_player == 1:
                self.time_black += seconds
            else:
                self.time_white += seconds
            
            # Manually update labels immediately
            self.lbl_black_time.setText(f"黑方: {self.time_black//60:02d}:{self.time_black%60:02d}")
            self.lbl_white_time.setText(f"白方: {self.time_white//60:02d}:{self.time_white%60:02d}")

            # Try to resume game if it was stopped due to timeout
            if self.time_black > 0 and self.time_white > 0:
                # Only resume if no one has won yet
                if self.check_winner() == 0:
                    if not self.game_active:
                        self.game_active = True
                        self.lbl_ai_status.setText("状态: 恢复 (时间已添加)")
                        # If it is AI's turn, trigger it
                        self.check_ai_turn()

    @staticmethod
    def get_expected_player(total_stones):
        """根据棋子总数计算当前应该是哪方下棋
        
        Args:
            total_stones: 当前棋盘上的总棋子数
            
        Returns:
            1 (黑方) 或 -1 (白方)
        """
        if total_stones == 0:
            return 1  # 黑方第一手
        elif total_stones == 1:
            return -1  # 白方
        elif total_stones == 2:
            return -1  # 白方第二子
        else:
            # 从第3子开始，每2子换一次
            # total=3,4 → 黑方, total=5,6 → 白方, total=7,8 → 黑方
            turn_index = (total_stones - 1) // 2
            return 1 if turn_index % 2 == 1 else -1

    def calculate_mark(self, win_rate):
        """
        根据胜率计算 MARK 值
        win_rate: 0.0 到 1.0
        返回: -2 (大劣), -1 (小劣), 0 (一般), 1 (小好), 2 (大好)
        """
        if win_rate < 0.20:
            return -2  # 大劣
        elif win_rate < 0.40:
            return -1  # 小劣
        elif win_rate < 0.60:
            return 0   # 一般
        elif win_rate < 0.80:
            return 1   # 小好
        else:
            return 2   # 大好
    
    def check_winner(self):
        """
        Check for 6+ in a row
        """
        board = self.board_widget.board
        rows, cols = board.shape
        
        # Directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for r in range(rows):
            for c in range(cols):
                player = board[r][c]
                if player == 0: continue
                
                for dr, dc in directions:
                    count = 0
                    for k in range(6): # Check 6 stones
                        nr, nc = r + k*dr, c + k*dc
                        if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == player:
                            count += 1
                        else:
                            break
                    if count >= 6:
                        return player
        return 0

    def update_turn_label(self):
        p_name = '黑方' if self.current_player == 1 else '白方'
        self.lbl_turn.setText(f"当前执子: {p_name} (剩余 {self.turn_moves_left} 子)")

    def update_timer(self):
        if not self.game_active or self.is_paused: return
        
        op = self.get_current_operator()
        
        # 团队模式人类回合倒计时
        # 必须是：1. 团队模式 2. 轮到我方 3. 操作者是人类（而非我方AI）
        my_color = 1 if self.team_is_black else -1
        is_my_turn_team = (self.current_player == my_color)
        
        if self.team_rotation_mode:
            if is_my_turn_team and op == 'Human':
                self.human_turn_timer -= 1
                self.update_notification_bar()
                
                # 超时判负 - 我方人类超时，对方获胜
                if self.human_turn_timer <= 0:
                    self.game_active = False
                    # 我方颜色超时，对方（-my_color）获胜
                    winner = -my_color  # 1=黑方胜, -1=白方胜
                    loser_name = "黑方" if my_color == 1 else "白方"
                    self.lbl_ai_status.setText(f"状态: {loser_name}超时判负")
                    self.show_game_result_dialog(winner)
                    return
            else:
                # 对方回合或AI回合，不减少 my_human_timer
                # 更新 Notification Bar 以保持正确状态显示
                self.update_notification_bar()
        
        # 更新双方总用时 (无论是 AI 还是人类，只要轮到该方，就扣时)
        if self.current_player == 1:
            self.time_black -= 1
        else:
            self.time_white -= 1
            
        self.lbl_black_time.setText(f"黑方: {self.time_black//60:02d}:{self.time_black%60:02d}")
        self.lbl_white_time.setText(f"白方: {self.time_white//60:02d}:{self.time_white%60:02d}")
        
        if self.time_black <= 0 or self.time_white <= 0:
            self.game_active = False
            # 时间耗尽，对方获胜
            if self.time_black <= 0:
                winner = -1  # 黑方超时，白方胜
                self.lbl_ai_status.setText("状态: 黑方时间耗尽")
            else:
                winner = 1   # 白方超时，黑方胜
                self.lbl_ai_status.setText("状态: 白方时间耗尽")
            self.show_game_result_dialog(winner)

    def handle_player_move(self, r, c):
        if not self.game_active or self.is_paused: return
        
        # STRICT Check: Is it Human's turn?
        if self.get_current_operator() != 'Human':
            print("❌ blocked human move during AI turn")
            return
        
        # 停止参考分析模式
        if self.reference_mode:
            self.reference_mode = False
            self.ai_worker.flush_commands()  # 取消正在进行的分析
        
        # Place Stone - 使用计算出的player而不是current_player
        expected_player = self.get_expected_player(len(self.moves))
        self.board_widget.board[r][c] = expected_player
        self.moves.append((r, c, expected_player))
        self.board_widget.last_move = (r, c)
        self.board_widget.update()
        
        # 同步current_player，确保与expected_player一致
        self.current_player = expected_player
        
        # Notify AI (Even if it's human move, AI needs to know to update its state)
        # 人类下棋时不需要 reexpand（AI 会在自己回合重新搜索）
        self.ai_worker.notify_move(r * 19 + c, is_same_turn_second=False)
        
        # Check Win
        winner = self.check_winner()
        if winner != 0:
            self.game_active = False
            self._clear_heatmap()
        # Show result directly
            self.show_game_result_dialog(winner)
            return

        # Check Draw (Board Full)
        if len(self.moves) >= BOARD_SIZE * BOARD_SIZE:
            self.game_active = False
            self._clear_heatmap()
            # Show result directly
            self.show_game_result_dialog(0)
            return

        # Logic
        self.turn_moves_left -= 1
        
        if self.turn_moves_left == 0:
            self.switch_turn()
        else:
            self.update_turn_label()

    def show_game_result_dialog(self, winner):
        """
        显示游戏结束对话框，让用户输入比赛信息并保存棋谱。
        :param winner: 1=黑胜, -1=白胜, 0=平局
        """
        dialog = GameResultDialog(winner, self)
        if dialog.exec_() == QDialog.Accepted:
            game_info = dialog.get_game_info()
            handler = C6SGFHandler()
            # 传递 move_evaluations 给保存函数
            saved_path = handler.save_game_with_info(game_info, self.moves, self.move_evaluations)
            if saved_path:
                QMessageBox.information(self, "保存成功", f"棋谱已保存到:\n{saved_path}")
            else:
                QMessageBox.warning(self, "保存失败", "棋谱保存失败，请检查文件权限。")

    def switch_turn(self):
        self.current_player = -self.current_player
        self.board_widget.current_player = self.current_player # Sync board player
        self.turn_moves_left = 2 # Always 2 after first
        self.update_turn_label()
        
        # 通知 AI 是否轮到对手（用于 Ponder）
        # 团队模式：对方颜色 = -my_color
        # 普通模式：对方颜色 = human_role (因为AI是对手)
        if self.team_rotation_mode:
            my_color = 1 if self.team_is_black else -1
            is_opponent_turn = (self.current_player != my_color)  # 不是我方 = 对方
        else:
            is_opponent_turn = (self.current_player == self.human_role)  # 人类回合 = 对手回合
        self.ai_worker.set_opponent_turn(is_opponent_turn)
        
        # 团队模式：重置人类回合计时器
        if self.team_rotation_mode:
            self.human_turn_timer = 30
        
        # Check AI Trigger
        self.check_ai_turn()
        
    def trigger_ai(self):
        if not self.game_active: return
        if self.is_paused: return
        if self.is_ai_thinking: return # Prevent spamming
        
        self.is_ai_thinking = True
        self.btn_undo.setEnabled(False)  # 禁用悔棋按钮
        # 轮到 AI 了，停止 Ponder
        self.ai_worker.set_opponent_turn(False)
        
        # === 状态校验 ===
        # 在 AI 思考前，校验 UI 和 AI 状态是否一致
        ui_board_flat = self.board_widget.board.flatten().tolist()
        self.ai_worker.request_state_verify(ui_board_flat, self.current_player)
        
        # === 动态设置 AI 颜色 (仅团队轮换模式需要) ===
        # 团队模式下 AI 可能在不同回合代表不同方，需要动态设置
        # 普通模式下 AI 颜色在 reset_game 时已固定，不需要再改
        if self.team_rotation_mode:
            print(f"🎯 [trigger_ai] 团队模式: set_ai_color({self.current_player})")
            self.ai_worker.set_ai_color(self.current_player)
        
        self.lbl_ai_status.setText("状态: AI 思考中...")
        self.btn_confirm_move.setText("确认落子 (Confirm Move)")
        self.board_widget.interaction_enabled = False
        
        # Prepare History
        hist = []
        for r, c, p in self.moves:
            hist.append(r * 19 + c) # MCTS expects 0-360
            
        self.ai_worker.request_move(hist, self.current_player, self.game_id)

    def on_ai_stats(self, stats):
        self.lbl_winrate.setText(f"胜率: {stats['win_rate']:.2f}")
        self.lbl_sims.setText(f"模拟数: {stats['sims']}")
        
        # 记录 AI 的胜率（用于 MARK 注释）
        self.last_ai_win_rate = stats['win_rate']
        
        # 更新热力图数据 (AI now sends (r, c, prob) format directly)
        if 'policy' in stats and stats['policy']:
            self.board_widget.policy_data = stats['policy']  # Direct assignment
            self.board_widget.update()
        
        # 显示思考信息
        time_str = f"{stats.get('time', 0):.1f}s"
        pruning_k = stats.get('pruning_k', 0)
        policy_count = len(stats.get('policy', []))
        self.lbl_debug_info.setText(f"Time={time_str} | K={pruning_k} | 候选={policy_count}")

    def on_ai_decision(self, r, c, game_id):
        # 竞态条件检查：如果这是旧对局的决策，直接丢弃
        if game_id != self.game_id:
            print(f"⚠️ [Ignored] 丢弃旧局决策 (MsgID: {game_id}, CurrID: {self.game_id})")
            return
        
        # 参考模式：只更新热力图，不落子，不显示具体坐标
        if self.reference_mode:
            # 热力图已经通过 on_ai_stats 更新，这里不执行任何落子逻辑
            # 不显示具体推荐坐标，避免被判作弊
            return
            
        self.is_ai_thinking = False # AI finished thinking this move
        self.btn_undo.setEnabled(True)  # 启用悔棋按钮
        
        # Reset button text
        self.btn_confirm_move.setText("确认落子 (Confirm Move)")
        
        # Check Debug Pause
        if self.chk_debug_pause.isChecked():
            self.pending_ai_move = (r, c)
            self.lbl_ai_status.setText(f"状态: AI 暂停 (等待确认 {format_move_coord(r, c)})")
            self.btn_confirm_move.setEnabled(True)
            return

        self.execute_ai_move(r, c)

    def execute_ai_move(self, r, c):
        self.lbl_ai_status.setText("状态: AI 落子")
        
        # Apply AI move - 使用计算出的player而不是current_player
        expected_player = self.get_expected_player(len(self.moves))
        self.board_widget.board[r][c] = expected_player
        self.moves.append((r, c, expected_player))
        self.board_widget.last_move = (r, c)
        self.board_widget.update()
        
        # 同步current_player，确保与expected_player一致
        self.current_player = expected_player
        
        # IMPORTANT: Notify AI thread that this move actually happened!
        # 检查下一子是否是同回合的第二子（用于子树复用优化）
        # turn_moves_left 在减 1 之前：2 表示还要下 2 子，1 表示还要下 1 子
        next_is_second_stone = (self.turn_moves_left == 2)  # 如果还剩 2 子，这是第一子，下一子需要 reexpand
        self.ai_worker.notify_move(r * 19 + c, is_same_turn_second=next_is_second_stone)
        
        # 记录 AI 着法的胜率评估（用于 MARK 注释）
        if self.last_ai_win_rate is not None and self.current_player != self.human_role:
            move_idx = len(self.moves) - 1  # 当前着法的索引
            mark = self.calculate_mark(self.last_ai_win_rate)
            if mark != 0:  # 只记录非零的 MARK
                self.move_evaluations[move_idx] = mark
        
        # Check Win
        winner = self.check_winner()
        if winner != 0:
            self.game_active = False
            self._clear_heatmap()
            self.lbl_ai_status.setText("状态: AI 胜利" if winner != self.human_role else "状态: 玩家胜利")
            # Show result directly
            self.show_game_result_dialog(winner)
            return
        
        # Check Draw (Board Full)
        if len(self.moves) >= BOARD_SIZE * BOARD_SIZE:
            self.game_active = False
            self._clear_heatmap()
            self.lbl_ai_status.setText("状态: 平局")
            self.show_game_result_dialog(0)
            return
        
        self.turn_moves_left -= 1
        if self.turn_moves_left == 0:
             self.switch_turn()
        else:
             self.update_turn_label()
             # If AI still has moves (e.g. 2nd stone), trigger again IMMEDIATELY
             # Don't wait for check_ai_turn's delay
             self.trigger_ai()

    def undo_move(self):
        """悔棋功能：每次撤销1子"""
        if not self.moves:
            return

        # 如果AI正在思考，不允许悔棋
        if self.is_ai_thinking:
            print("⚠️ [Undo] AI正在思考，请等待...")
            return

        # 停止 AI 触发定时器
        self.ai_trigger_timer.stop()
        self.ai_worker.flush_commands()

        # === 简单策略：每次只撤销1子 ===
        # 从棋盘读取实际颜色（更可靠）
        last_move = self.moves[-1]
        r, c = last_move[0], last_move[1]
        actual_color = self.board_widget.board[r][c]
        
        # 移除棋子
        self.moves.pop()
        self.board_widget.board[r][c] = 0
        
        # 同步AI状态：重置并重放所有着法
        self.ai_worker.reset_game()
        for mr, mc, mp in self.moves:
            self.ai_worker.notify_move(mr * 19 + mc)
        

        
        # 根据剩余棋子数重新计算状态
        total = len(self.moves)
        
        if total == 0:
            self.current_player = 1
            self.turn_moves_left = 1
        elif total == 1:
            self.current_player = -1
            self.turn_moves_left = 2
        elif total == 2:
            self.current_player = -1
            self.turn_moves_left = 1
        else:
            # Use the robust helper method
            self.current_player = self.get_expected_player(total)
            
            # Calculate stones left in current turn
            # Total 1 (B) -> Next W1 (Start of turn, Left 2)
            # Total 2 (B, W) -> Next W2 (Mid turn, Left 1)
            # Total 3 (B, W, W) -> Next B1 (Start, Left 2)
            # Pattern: if (total - 1) is even -> Start of turn (2 left). Odd -> Mid turn (1 left).
            is_start_of_turn = ((total - 1) % 2 == 0)
            self.turn_moves_left = 2 if is_start_of_turn else 1
        
        # 更新UI
        self.board_widget.current_player = self.current_player
        self.board_widget.update()
        self.update_turn_label()
        
        # 只有当操作者是人类且还有剩余子数时才允许交互
        # 使用 get_current_operator() 支持团队模式（对手也是人类）
        op = self.get_current_operator()
        has_moves_left = (self.turn_moves_left > 0)
        self.board_widget.interaction_enabled = (op == 'Human' and has_moves_left)
        
        # 重置人类回合计时器（团队模式）
        if self.team_rotation_mode:
            self.human_turn_timer = 30
        
        # 更新操作者标签和通知栏
        self.update_operator_label()
        
        self.lbl_ai_status.setText("状态: 悔棋完成，等待落子")
        self.is_paused = False
        self.game_active = True
        
        # 输出悔棋信息
        player_name = '黑方' if self.current_player == 1 else '白方'
        stone_num = 3 - self.turn_moves_left
        print(f"✅ [Undo] 悔棋成功：撤销1子，剩余{total}子，轮到{player_name}下第{stone_num}子")
        print(f"🔍 [Undo] current_player={self.current_player}, turn_moves_left={self.turn_moves_left}, human_role={self.human_role}")
        
        # 如果悔棋后turn_moves_left=0，说明这回合已下完
        # 需要手动处理回合切换
        if self.turn_moves_left == 0:
            # 切换到下一方
            self.current_player = -self.current_player
            self.board_widget.current_player = self.current_player
            self.turn_moves_left = 2
            self.update_turn_label()
            
            # 更新交互状态 - 使用 get_current_operator() 支持团队模式
            op = self.get_current_operator()
            self.board_widget.interaction_enabled = (op == 'Human')
            
            # 如果是AI回合，延迟1秒后触发AI（给用户时间继续悔棋）
            if op == 'AI':
                self.lbl_ai_status.setText("状态: 悔棋完成，1秒后AI思考...")
                self.ai_trigger_timer.start(1000)  # 1秒延迟
        
        # 更新悔棋按钮状态：只要有棋子就可以悔棋
        self.btn_undo.setEnabled(len(self.moves) > 0)


    def save_game(self):
        """手动保存棋谱，对局中保存显示为流局"""
        if not self.moves:
            QMessageBox.warning(self, "无法保存", "当前没有棋谱可以保存。")
            return
        
        # 使用 winner=2 表示流局（对局中保存）
        self.show_game_result_dialog(2)

    def load_game(self):
        handler = C6SGFHandler()
        # 默认打开 assets 目录
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
        if not os.path.exists(default_dir):
            default_dir = ""
        fname, _ = QFileDialog.getOpenFileName(self, "载入棋谱", default_dir, "Text Files (*.txt)")
        if fname:
            loaded_moves = handler.load_game(fname)
            if not loaded_moves:
                QMessageBox.warning(self, "载入失败", "无法解析棋谱文件，请检查文件格式。")
                return
            
            # === 询问用户执黑还是执白 ===
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QPushButton, QLabel
            
            role_dialog = QDialog(self)
            role_dialog.setWindowTitle("选择角色")
            role_dialog.setMinimumWidth(350)
            
            layout = QVBoxLayout(role_dialog)
            layout.setSpacing(15)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # 标题
            title_label = QLabel("请选择 AI 的阵营：")
            title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e0e0e0;")
            layout.addWidget(title_label)
            
            # 单选按钮组
            button_group = QButtonGroup(role_dialog)
            
            radio_black = QRadioButton(" AI 执黑 (先手)")
            radio_black.setStyleSheet("""
                QRadioButton {
                    font-size: 14px;
                    color: #ffffff;
                    padding: 10px;
                    background-color: #333333;
                    border-radius: 6px;
                    margin: 5px 0;
                }
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                }
                QRadioButton:hover {
                    background-color: #3e3e42;
                }
            """)
            radio_black.setChecked(True)  # 默认选择黑方(AI)
            button_group.addButton(radio_black)
            button_group.setId(radio_black, 1)  # 设置 ID 为 1（AI执黑）
            layout.addWidget(radio_black)
            
            radio_white = QRadioButton(" AI 执白 (后手)")
            radio_white.setStyleSheet("""
                QRadioButton {
                    font-size: 14px;
                    color: #ffffff;
                    padding: 10px;
                    background-color: #333333;
                    border-radius: 6px;
                    margin: 5px 0;
                }
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                }
                QRadioButton:hover {
                    background-color: #3e3e42;
                }
            """)
            button_group.addButton(radio_white)
            button_group.setId(radio_white, -1)  # 设置 ID 为 -1（AI执白）
            layout.addWidget(radio_white)
            
            # 按钮
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            
            btn_ok = QPushButton("确定")
            btn_ok.setStyleSheet("""
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: none;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1f8ad2;
                }
            """)
            btn_ok.clicked.connect(role_dialog.accept)
            btn_layout.addWidget(btn_ok)
            
            layout.addLayout(btn_layout)
            
            # 设置对话框样式
            role_dialog.setStyleSheet("""
                QDialog {
                    background-color: #252526;
                    color: #e0e0e0;
                }
            """)
            
            # 显示对话框并获取结果
            if role_dialog.exec_() != QDialog.Accepted:
                return  # 用户取消
            
            # 获取用户选择的角色（直接检查哪个按钮被选中，避免 ID 问题）
            # 获取用户选择的角色 (AI role)
            # 按钮已经改成了 "AI 执黑/白"
            if radio_black.isChecked():
                ai_side = 1  # AI黑方
            elif radio_white.isChecked():
                ai_side = -1  # AI白方
            else:
                ai_side = 1  # 默认AI黑方
            
            print(f"🎭 [Load] User selected AI side: {ai_side} ({'Black' if ai_side == 1 else 'White'})")
            
            # === 完整重置游戏状态 ===
            self.moves = []
            self.board_widget.board.fill(0)
            self.current_player = 1
            self.turn_moves_left = 1
            self.is_ai_thinking = False
            
            # 重置评估相关状态
            self.move_evaluations = {}  # 清空着法评估
            self.last_ai_win_rate = None  # 重置胜率记录
            
            # 重置时间
            self.time_black = 600
            self.time_white = 600
            
            # 增加 Game ID（新的棋谱加载视为新对局）
            self.game_id += 1
            print(f"📂 [Load] Loading game record (ID: {self.game_id})")
            
            # 重置 AI 状态（同步方式：直接操作 move_history）
            self.ai_worker.flush_commands()
            self.ai_worker.reset_game()
            
            # 等待 RESET 命令被处理（给一点时间让队列处理）
            import time
            time.sleep(0.1)
            
            # 收集所有着法的 move_idx，直接设置 AI 的 move_history
            all_move_indices = []
            
            # 重放所有着法到 UI
            current_p = 1
            stones_in_turn = 0
            total_stones = 0
            
            for i, (r, c, p) in enumerate(loaded_moves):
                # 使用计算出的player而不是current_p
                expected_player = self.get_expected_player(len(self.moves))
                self.board_widget.board[r][c] = expected_player
                self.moves.append((r, c, expected_player))
                all_move_indices.append(r * 19 + c)
                
                total_stones += 1
                stones_in_turn += 1
                
                # Connect6 规则：黑方第一手下 1 子，之后每方下 2 子
                if total_stones == 1:
                    # 黑方第一手结束
                    current_p = -1
                    stones_in_turn = 0
                elif stones_in_turn >= 2:
                    # 当前方下完 2 子，换边
                    current_p = -current_p
                    stones_in_turn = 0
            
            # 直接同步 AI 状态（不通过队列，避免异步问题）
            if self.ai_worker.mcts:
                self.ai_worker.move_history = all_move_indices.copy()
                self.ai_worker.mcts.sync_state_from_moves(all_move_indices)
                self.ai_worker._reset_ponder_state()
                self.ai_worker.opponent_turn = False
                self.ai_worker.opponent_stones_in_turn = 0
            
            self.current_player = current_p
            self.board_widget.current_player = current_p
            
            # 计算剩余子数
            if total_stones == 0:
                self.turn_moves_left = 1
            elif stones_in_turn == 0:
                # 刚换边，需要下 2 子（除非是黑方第一手后）
                self.turn_moves_left = 2 if total_stones > 1 else 1
            else:
                # 在回合中间，还需要下 1 子
                self.turn_moves_left = 2 - stones_in_turn
            
            # 更新最后一步标记
            if self.moves:
                self.board_widget.last_move = (self.moves[-1][0], self.moves[-1][1])
            
            # === 关键修复：加载后暂停游戏，等待用户操作 ===
            self.game_active = True
            self.is_paused = True  # 暂停状态，防止自动触发 AI 或 Ponder
            self.btn_pause.setText("继续 (Resume)")
            self.board_widget.interaction_enabled = False  # 禁用交互，直到用户点击继续
            
            # === 根据用户选择设置角色 ===
            # ai_side 是用户选的 AI 颜色
            self.human_role = -ai_side
            print(f"✅ [Load] Derived human_role: {self.human_role} (Opposite of AI)")
            
            # 更新 combo box 显示（阻止信号避免触发 on_player_side_change）
            # combo_side index 0 = AI Black(1), index 1 = AI White(-1)
            combo_index = 0 if ai_side == 1 else 1
            self.combo_player.blockSignals(True)
            self.combo_player.setCurrentIndex(combo_index)
            self.combo_player.blockSignals(False)
            
            # 设置 AI 颜色
            ai_color = ai_side
            self.ai_worker.set_ai_color(ai_color)
            
            # === 关键修复：同步 Team Mode 状态 ===
            # 如果当前处于团队模式，必须更新 team_is_black 以匹配选择的 AI 阵营
            # 否则 get_current_operator 会基于旧状态判断错误
            self.team_is_black = (ai_side == 1)
            print(f"✅ [Load] Updated team_is_black: {self.team_is_black}")
            
            # 不要在加载后立即触发 AI 或 Ponder
            # 用户需要先点击「继续」按钮
            self.ai_worker.set_opponent_turn(False)  # 关闭 Ponder
            
            self.board_widget.update()
            self.update_turn_label()
            
            # 更新时间显示
            self.lbl_black_time.setText(f"黑方: {self.time_black//60:02d}:{self.time_black%60:02d}")
            self.lbl_white_time.setText(f"白方: {self.time_white//60:02d}:{self.time_white%60:02d}")
            
            # 更新状态显示
            player_name = '黑方' if self.current_player == 1 else '白方'
            self.lbl_ai_status.setText(f"状态: 已载入棋谱 ({len(self.moves)} 步) - 点击「继续」")
            
            # 更新悔棋按钮状态：加载后应该可以悔棋
            self.btn_undo.setEnabled(len(self.moves) > 0)
            
            # 使用美化的消息框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("载入成功")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(f"✅ 成功载入棋谱，共 {len(self.moves)} 步")
            msg_box.setInformativeText(f"当前轮到{player_name}下棋。\n\n点击「继续」按钮开始对局，或点击「悔棋」回退。")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #252526;
                    color: #e0e0e0;
                }
                QMessageBox QLabel {
                    color: #e0e0e0;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #0d6efd;
                    color: white;
                    border: none;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #0b5ed7;
                }
            """)
            msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
