# Gomoku AI Project

## 概要
Webカメラで盤面を認識し、ガントリーロボットを使用して五目並べ（連珠）を行うAIシステムです。

## ディレクトリ構成
- **src/**: メインのソースコード
  - `main.py`: メイン実行ファイル
  - `gomoku_board_recognition.py`: 画像認識ロジック
  - `cpp_extension/`: C++による高速AI（開発中）
- **tools/**: ユーティリティツール
  - `flowchart_generator.py`: フローチャート生成など
- **tests/**: テスト用コード
  - `AIvsAI/`: AI同士の対戦シミュレーション
- **legacy/**: 過去のバージョンや旧コード
- **docs/**: ドキュメント、参考画像
- **hub.py**: アプリケーション起動ランチャー

## 実行方法
ルートディレクトリで以下のコマンドを実行し、メニューから起動したいモードを選択してください。

```powershell
python hub.py
```
