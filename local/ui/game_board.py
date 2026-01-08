import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PyQt5.QtCore import Qt, pyqtSignal

# Constants
BOARD_SIZE = 19
MARGIN = 40

def format_move_coord(r, c):
    """
    将内部坐标 (r, c) 转换为标准棋盘坐标字符串。
    内部: r=0 是顶部, r=18 是底部
    显示: 行号从下往上 1-19, 列字母 A-S
    例如: (0, 9) -> "J19", (18, 0) -> "A1 " (填充到3字符)
    """
    col_char = chr(ord('A') + c)
    row_num = BOARD_SIZE - r  # r=0 -> 19, r=18 -> 1
    # 填充到3字符，避免 UI 抖动 (L9 -> "L9 ")
    return f"{col_char}{row_num:<2}"

class Connect6Board(QWidget):
    move_signal = pyqtSignal(int, int) # r, c

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.last_move = None
        self.hover_pos = None
        self.setMouseTracking(True)
        self.interaction_enabled = True
        
        # Dynamic resizing vars
        self.grid_size = 30
        self.offset_x = MARGIN
        self.offset_y = MARGIN
        self.setMinimumSize(400, 400)
        
        # Visualization State
        self.show_policy = False
        self.policy_data = []  # List of (r, c, prob)
        self.current_player = 1  # Default to Black

    def set_board(self, board, last_move=None):
        self.board = board
        self.last_move = last_move
        self.update()
        
    def resizeEvent(self, event):
        # Calculate grid size to fit in the widget
        w = self.width()
        h = self.height()
        # Reserve space for labels (MARGIN) on all sides
        # Grid itself needs (BOARD_SIZE - 1) * unit
        # Available size = Min(w, h) - 2 * MARGIN
        available_size = min(w, h) - 1.5 * MARGIN
        self.grid_size = available_size / (BOARD_SIZE - 1)
        
        # Center the board
        self.offset_x = (w - (BOARD_SIZE - 1) * self.grid_size) / 2
        self.offset_y = (h - (BOARD_SIZE - 1) * self.grid_size) / 2
        self.update()

    def mouseMoveEvent(self, event):
        if not self.interaction_enabled: return
        
        col = round((event.x() - self.offset_x) / self.grid_size)
        row = round((event.y() - self.offset_y) / self.grid_size)
        
        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
            self.hover_pos = (row, col)
        else:
            self.hover_pos = None
        self.update()

    def mousePressEvent(self, event):
        if not self.interaction_enabled: return
        if event.button() == Qt.LeftButton and self.hover_pos:
            r, c = self.hover_pos
            if self.board[r][c] == 0:
                self.move_signal.emit(r, c)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw Background (Dark Grey everywhere)
        painter.fillRect(self.rect(), QColor(30, 30, 30)) 
        
        # Draw Wood Texture Background
        # Fill the entire board area including margins
        board_left = self.offset_x - MARGIN
        board_top = self.offset_y - MARGIN
        board_w = (BOARD_SIZE - 1) * self.grid_size + 2 * MARGIN
        board_h = (BOARD_SIZE - 1) * self.grid_size + 2 * MARGIN
        
        painter.fillRect(int(board_left), int(board_top), int(board_w), int(board_h), QColor(220, 179, 92))
        
        # Draw Border Line
        painter.setPen(QPen(QColor(100, 70, 20), 2))
        painter.drawRect(int(board_left), int(board_top), int(board_w), int(board_h))

        # Draw Grid
        pen = QPen(Qt.black, 1)
        painter.setPen(pen)
        
        # Labels font (Small, Black, on Wood)
        # Font size relative to grid size, small enough to fit in margin
        font_size = min(14, int(self.grid_size * 0.45)) 
        font = QFont("Arial", font_size)
        font.setBold(True)
        painter.setFont(font) 
        
        for i in range(BOARD_SIZE):
            # Horizontal lines
            y = self.offset_y + i * self.grid_size
            painter.drawLine(int(self.offset_x), int(y), 
                             int(self.offset_x + (BOARD_SIZE - 1) * self.grid_size), int(y))
            # Vertical lines
            x = self.offset_x + i * self.grid_size
            painter.drawLine(int(x), int(self.offset_y), 
                             int(x), int(self.offset_y + (BOARD_SIZE - 1) * self.grid_size))
                             
            # Draw Labels INSIDE the wood board
            painter.setPen(Qt.black) # Black text on wood
            text_flags = Qt.AlignCenter
            
            # Row numbers (19-1) on LEFT side
            row_label = str(BOARD_SIZE - i)
            # Position: In the margin between wood edge and first grid line
            # Center it in that margin space
            label_rect_x = int(self.offset_x - MARGIN)
            label_rect_w = int(MARGIN)
            label_rect_y = int(y - self.grid_size/2)
            painter.drawText(label_rect_x, label_rect_y, label_rect_w, int(self.grid_size), text_flags, row_label)
            
            # Col letters (A-S) on BOTTOM side
            col_char = chr(ord('A') + i)
            # Position: In the margin below last grid line
            label_rect_x = int(x - self.grid_size/2)
            label_rect_y = int(self.offset_y + (BOARD_SIZE - 1) * self.grid_size)
            label_rect_h = int(MARGIN)
            painter.drawText(label_rect_x, label_rect_y, int(self.grid_size), label_rect_h, text_flags, col_char)
            
            # Reset pen for grid
            painter.setPen(pen)

        # Draw Star Points
        stars = [3, 9, 15]
        painter.setBrush(Qt.black)
        star_rad = self.grid_size * 0.15
        for r in stars:
            for c in stars:
                cx = self.offset_x + c * self.grid_size
                cy = self.offset_y + r * self.grid_size
                painter.drawEllipse(int(cx - star_rad), int(cy - star_rad), int(star_rad*2), int(star_rad*2))

        # Draw Policy Heatmap (If enabled)
        if self.show_policy and self.policy_data:
            # policy_data is a list of (r, c, prob)
            max_prob = 0.0
            if self.policy_data:
                max_prob = max(p for _, _, p in self.policy_data)
            
            # Draw ALL policy points, even if prob is 0 (epsilon)
            # We assume policy_data only contains relevant points (children of root)
            if self.policy_data:
                # Find max for color scaling, but ensure we don't divide by zero
                max_prob = max(p for _, _, p in self.policy_data)
                if max_prob < 1e-9: max_prob = 1.0 

                for r, c, prob in self.policy_data:
                    # No filtering! Show everything sent by AI.
                    
                    cx = self.offset_x + c * self.grid_size
                    cy = self.offset_y + r * self.grid_size
                    
                    # Color based on prob relative to max
                    # Red = High, Blue = Low
                    ratio = prob / max_prob
                    # Simple alpha blend: Red with alpha
                    alpha = int(150 * ratio + 50) # 50-200
                    color = QColor(255, 0, 0, alpha)
                    
                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    
                    # Draw square
                    rect_size = self.grid_size * 0.8
                    rect_x = int(cx - rect_size/2)
                    rect_y = int(cy - rect_size/2)
                    painter.drawRect(rect_x, rect_y, int(rect_size), int(rect_size))
                    
                    # Draw prob text (Centered and High Contrast)
                    painter.setPen(Qt.white)
                    font_sz = max(8, int(self.grid_size * 0.4))
                    font = QFont("Arial", font_sz)
                    font.setBold(True)
                    painter.setFont(font)
                    
                    prob_txt = f"{int(prob*100)}"
                    
                    # Draw text outline (Black shadow) for readability
                    path = QPainterPath()
                    path.addText(rect_x + rect_size/2 - font_sz/1.5, rect_y + rect_size/2 + font_sz/3, font, prob_txt)
                    
                    # Simplified text drawing: Draw black text slightly offset, then white text
                    # Actually, just drawing it centered in the rect is enough if we use a good color
                    # Let's use Yellow for high prob, White for low
                    
                    text_color = Qt.yellow if prob > 0.1 else Qt.white
                    
                    # Draw Shadow/Outline
                    painter.setPen(Qt.black)
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                        painter.drawText(rect_x + dx, rect_y + dy, int(rect_size), int(rect_size), Qt.AlignCenter, prob_txt)
                        
                    # Draw Main Text
                    painter.setPen(text_color)
                    painter.drawText(rect_x, rect_y, int(rect_size), int(rect_size), Qt.AlignCenter, prob_txt)

        # Draw Stones
        stone_rad = self.grid_size * 0.45
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != 0:
                    cx = self.offset_x + c * self.grid_size
                    cy = self.offset_y + r * self.grid_size
                    
                    if self.board[r][c] == 1: # Black
                        painter.setBrush(Qt.black)
                        painter.setPen(Qt.NoPen)
                    else: # White
                        painter.setBrush(Qt.white)
                        painter.setPen(Qt.black)
                        
                    painter.drawEllipse(int(cx - stone_rad), int(cy - stone_rad), int(stone_rad*2), int(stone_rad*2))
                    
                    # Highlight Last Move
                    if self.last_move == (r, c):
                        painter.setBrush(Qt.red)
                        painter.drawEllipse(int(cx - star_rad), int(cy - star_rad), int(star_rad*2), int(star_rad*2))

        # Draw Hover (Ghost Stone)
        if self.hover_pos and self.interaction_enabled:
            r, c = self.hover_pos
            if self.board[r][c] == 0:
                cx = self.offset_x + c * self.grid_size
                cy = self.offset_y + r * self.grid_size
                
                if self.current_player == 1: # Black
                    painter.setBrush(QColor(0, 0, 0, 100)) # Semi-transparent black
                else: # White
                    painter.setBrush(QColor(255, 255, 255, 150)) # Semi-transparent white (slightly more opaque)
                    
                painter.drawEllipse(int(cx - stone_rad*0.8), int(cy - stone_rad*0.8), int(stone_rad*1.6), int(stone_rad*1.6))
