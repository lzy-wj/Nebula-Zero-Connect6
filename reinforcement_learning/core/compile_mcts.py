"""
编译 MCTS 引擎 (跨平台支持 Windows/Linux)
使用方法: python compile_mcts.py
"""
import subprocess
import os
import sys
import shutil
import platform

# 切换到脚本所在目录 (core/)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def compile_windows():
    """Windows: 使用 MSVC (cl.exe)"""
    if not shutil.which("cl"):
        print("❌ 错误: 找不到 cl.exe")
        print("请从 'x64 Native Tools Command Prompt for VS 2022' 运行此脚本")
        return 1
    
    print("🔨 [Windows] 编译 mcts_engine.cpp...")
    
    cmd = 'cl /utf-8 /LD /std:c++17 /EHsc /O2 /openmp mcts_engine.cpp /I . /Fe:mcts.dll'
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    if os.path.exists("mcts.dll"):
        print("✅ 编译成功: mcts.dll")
    else:
        print("❌ 找不到编译输出的 mcts.dll")
        return 1
    
    # 清理中间文件
    for f in ["mcts.obj", "mcts_engine.obj", "mcts.exp", "mcts.lib"]:
        if os.path.exists(f):
            os.remove(f)
    
    return 0

def compile_linux():
    """Linux: 使用 g++"""
    if not shutil.which("g++"):
        print("❌ 错误: 找不到 g++")
        print("请安装 g++: sudo apt install g++")
        return 1
    
    print("🔨 [Linux] 编译 mcts_engine.cpp...")
    
    cmd = 'g++ -shared -fPIC -O3 -fopenmp -std=c++17 -I. mcts_engine.cpp -o libmcts.so'
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    if os.path.exists("libmcts.so"):
        print("✅ 编译成功: libmcts.so")
    else:
        print("❌ 找不到编译输出的 libmcts.so")
        return 1
    
    return 0

def compile():
    system = platform.system()
    
    if system == "Windows":
        return compile_windows()
    elif system == "Linux":
        return compile_linux()
    else:
        print(f"❌ 不支持的操作系统: {system}")
        return 1

if __name__ == "__main__":
    sys.exit(compile())
