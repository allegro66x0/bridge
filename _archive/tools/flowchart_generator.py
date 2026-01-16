import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz\bin'
from graphviz import Digraph

def create_flowchart():
    # グラフの作成
    dot = Digraph(comment='Gantry Control Flowchart', format='png')
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr('node', fontname='Meiryo')  # 日本語フォント指定（環境に合わせて変更してください ex: 'MS Gothic', 'Hiragino Sans'）

    # --- ノードの定義 ---
    
    # 開始
    dot.node('Start', '開始 / 電源ON', shape='oval', style='filled', fillcolor='lightgrey')

    # Setup Phase (サブグラフ)
    with dot.subgraph(name='cluster_Setup') as c:
        c.attr(label='セットアップ (setup)', fontname='Meiryo', style='dashed')
        c.node('SerialInit', 'シリアル通信開始\n9600bps', shape='box')
        c.node('PinSetup', 'ピンモード設定\n& モーター設定', shape='box')
        c.node('InitHome', '初期原点復帰\n(homeAxis X & Y)', shape='box')
        c.node('SendReady1', "'READY' 送信", shape='parallelogram', style='filled', fillcolor='lightblue')
        
        c.edge('SerialInit', 'PinSetup')
        c.edge('PinSetup', 'InitHome')
        c.edge('InitHome', 'SendReady1')

    # Loop Phase (サブグラフ)
    with dot.subgraph(name='cluster_Loop') as c:
        c.attr(label='メインループ (loop)', fontname='Meiryo', style='dashed')
        
        c.node('LoopStart', 'シリアル受信\nあり?', shape='diamond', style='filled', fillcolor='lightyellow')
        c.node('ReadChar', '1文字読み込み', shape='box')
        c.node('IsDigit', '数値か?', shape='diamond', style='filled', fillcolor='lightyellow')
        c.node('AddBuffer', 'バッファに追加', shape='box')
        c.node('CheckNewline', "改行コード'\\n' か?", shape='diamond', style='filled', fillcolor='lightyellow')
        c.node('CheckLen', 'バッファ長\n== 4?', shape='diamond', style='filled', fillcolor='lightyellow')
        c.node('ResetBuf', 'バッファリセット', shape='box')
        c.node('ParseCoords', 'X, Y 座標\n文字列の解析', shape='box')

        # 移動シーケンス (さらに内部のサブグラフ)
        with c.subgraph(name='cluster_MoveSeq') as m:
            m.attr(label='移動シーケンス\n(executeMoveSequence)', fontname='Meiryo', style='solid')
            m.node('CalcSteps', '目標ステップ数計算', shape='box')
            m.node('MoveCmd', '目標位置へ\n移動開始', shape='box')
            m.node('WaitMov', '移動完了待ち', shape='box')
            m.node('ActionDelay', '待機 1000ms\n(コマ置き動作)', shape='box')
            m.node('ReHome', '再原点復帰\n(位置ズレ補正)', shape='box')
            
            m.edge('CalcSteps', 'MoveCmd')
            m.edge('MoveCmd', 'WaitMov')
            m.edge('WaitMov', 'ActionDelay')
            m.edge('ActionDelay', 'ReHome')

        c.node('SendReady2', "'READY' 送信", shape='parallelogram', style='filled', fillcolor='lightblue')

        # Loop内のエッジ接続
        c.edge('LoopStart', 'LoopStart', label='No')
        c.edge('LoopStart', 'ReadChar', label='Yes')
        c.edge('ReadChar', 'IsDigit')
        
        c.edge('IsDigit', 'AddBuffer', label='Yes')
        c.edge('IsDigit', 'CheckNewline', label='No')
        c.edge('AddBuffer', 'CheckNewline')
        
        c.edge('CheckNewline', 'LoopStart', label='No')
        c.edge('CheckNewline', 'CheckLen', label='Yes')
        
        c.edge('CheckLen', 'ResetBuf', label='No')
        c.edge('CheckLen', 'ParseCoords', label='Yes')
        
        c.edge('ParseCoords', 'CalcSteps')
        c.edge('ReHome', 'SendReady2')
        c.edge('SendReady2', 'ResetBuf')
        c.edge('ResetBuf', 'LoopStart')

    # 全体の接続
    dot.edge('Start', 'SerialInit')
    dot.edge('SendReady1', 'LoopStart')

    # 保存とレンダリング
    output_filename = 'gantry_flowchart'
    dot.render(output_filename, cleanup=True)
    print(f"画像ファイルを生成しました: {output_filename}.png")

if __name__ == '__main__':
    create_flowchart()