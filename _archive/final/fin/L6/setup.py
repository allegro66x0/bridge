import sys
from setuptools import setup, Extension
import pybind11

# 警告: g++ (MinGW) の場合、OpenMPやstd::thread関連で問題が起きやすいため、
# C++の並列化機能を無効にするコンパイルフラグを推奨。
# (Python側で並列化する方が安全な場合がある)
# しかし、ここではC++スレッドを有効にするため '-pthread' を追加します。

# C++のコンパイラフラグ
# /O2 (MSVC) または -O3 (g++) は最適化
cpp_args = ['-std=c++17', '-O3', '-pthread']
if sys.platform == 'win32':
    cpp_args = ['/std:c++17', '/O2']


# Pythonモジュールを定義
ext_modules = [
    Extension(
        'cpp_gomoku_ai', # 生成されるモジュール名 (import cpp_gomoku_ai)
        [
            'gomoku_ai.cpp',       # AIロジック本体
            'pybind_wrapper.cpp' # Pythonとの通訳
        ],
        include_dirs=[
            pybind11.get_include(), # Pybind11のヘッダー
        ],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=['-pthread'] if sys.platform != 'win32' else []
    ),
]

setup(
    name='cpp_gomoku_ai',
    version='1.0',
    description='Gomoku AI C++ engine',
    ext_modules=ext_modules,
)