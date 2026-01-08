import os
import datetime

class GameRecorder:
    def __init__(self):
        self.header = {
            "GameName": "Connect6",
            "BlackTeam": "Unknown",
            "WhiteTeam": "Unknown",
            "Result": "Unknown", # "BlackWin", "WhiteWin", "Draw"
            "Time": datetime.datetime.now().strftime("%Y.%m.%d %H:%M"),
            "Place": "Competition",
            "Event": "2025 C6 Cup"
        }
        self.moves = [] # List of (color_char, coord_str, comment)
        # coord_str like "J10"
    
    def set_header(self, key, value):
        self.header[key] = value
        
    def add_move(self, color, x, y, comment=""):
        """
        color: 'B' or 'W'
        x: 0-18 (A-S)
        y: 0-18 (1-19)
        """
        col_char = chr(ord('A') + x)
        row_str = str(y + 1)
        coord = f"{col_char},{row_str}"
        self.moves.append({"c": color, "p": coord, "m": comment})

    def save(self, directory):
        # Filename format: C6-Black vs White-Result-Time-Event.txt
        # Time in filename usually compact? The example shows full format in content, 
        # but filename usually simpler. Let's follow the example pattern if explicit, 
        # otherwise sensible default.
        # Example filename: C6-先手参赛队B vs 后手参赛队W-先(后)手胜.txt
        
        h = self.header
        result_short = h["Result"]
        if "Black" in result_short: result_short = "先手胜"
        elif "White" in result_short: result_short = "后手胜"
        else: result_short = "平局"
        
        filename = f"C6-{h['BlackTeam']} vs {h['WhiteTeam']}-{result_short}-{h['Event']}.txt"
        # Sanitize filename
        filename = filename.replace(":", "").replace("/", "-")
        
        path = os.path.join(directory, filename)
        
        # Content generation
        # {[C6][Black][White][Result][Time][Place][Event];B(J,10);W(I,11)...}
        
        content = "{"
        content += f"[C6][{h['BlackTeam']}][{h['WhiteTeam']}][{h['Result']}][{h['Time']}][{h['Place']}][{h['Event']}]"
        
        for move in self.moves:
            # Format: ;B(J,10) or ;B(J,10)MARK[1]
            # Note: Example uses parentheses B(J,10)
            entry = f";{move['c']}({move['p']})"
            if move['m']:
                entry += f"MARK[{move['m']}]" # Simple mapping for now
            content += entry
            
        content += "}"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return path
