"""
编译 MCTS 引擎 (跨平台支持 Windows/Linux)
使用方法: python compile_mcts.py
"""
import subprocess
import os
import sys
import shutil
import platform
import argparse

# 切换到脚本所在目录 (core/)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def compile_windows(output):
    """Windows: 使用 MSVC (cl.exe)"""
    if not shutil.which("cl"):
        print("❌ 错误: 找不到 cl.exe")
        print("请从 'x64 Native Tools Command Prompt for VS 2022' 运行此脚本")
        return 1
    
    print("🔨 [Windows] 编译 mcts_engine.cpp...")
    
    temporary_output = f"{output}.tmp"
    cmd = (
        'cl /utf-8 /LD /std:c++17 /EHsc /O2 /openmp '
        f'mcts_engine.cpp /I . /Fe:{temporary_output}'
    )
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    if os.path.exists(temporary_output):
        os.replace(temporary_output, output)
        print(f"✅ 编译成功: {output}")
    else:
        print("❌ 找不到编译输出的 mcts.dll")
        return 1
    
    # 清理中间文件
    for f in ["mcts.obj", "mcts_engine.obj", "mcts.exp", "mcts.lib"]:
        if os.path.exists(f):
            os.remove(f)
    
    return 0

def compile_linux(output):
    """Linux: 使用 g++"""
    if not shutil.which("g++"):
        print("❌ 错误: 找不到 g++")
        print("请安装 g++: sudo apt install g++")
        return 1
    
    print("🔨 [Linux] 编译 mcts_engine.cpp...")
    
    output = os.path.abspath(output)
    temporary_output = f"{output}.tmp.{os.getpid()}"
    cmd = [
        'g++', '-shared', '-fPIC', '-O3', '-fopenmp', '-std=c++17',
    ]
    # 本地和远端都会在目标服务器上从源码编译。默认针对当前 CPU 生成
    # AVX2/F16C 等指令，第二手低秩打分可直接走向量化路径；如需制作
    # 可跨不同 CPU 搬运的通用动态库，可显式设置 NEBULA_MCTS_NATIVE=0。
    if os.environ.get('NEBULA_MCTS_NATIVE', '1') == '1':
        cmd.append('-march=native')
    cmd.extend([
        '-I.', 'mcts_engine.cpp', '-o', temporary_output,
    ])
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    if os.path.exists(temporary_output):
        os.replace(temporary_output, output)
        print(f"✅ 编译成功: {output}")
    else:
        print("❌ 找不到编译输出的 libmcts.so")
        return 1
    
    return 0

def compile(output):
    system = platform.system()
    
    if system == "Windows":
        return compile_windows(output)
    elif system == "Linux":
        return compile_linux(output)
    else:
        print(f"❌ 不支持的操作系统: {system}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="编译 MCTS 动态库")
    parser.add_argument(
        '--output',
        default='mcts.dll' if platform.system() == 'Windows' else 'libmcts.so',
        help='输出动态库路径；先写临时文件再原子替换',
    )
    arguments = parser.parse_args()
    sys.exit(compile(arguments.output))
