#!/usr/bin/env python
# -*- coding: utf-8 -*-

def decode_bytes_sample():
    """バイト文字列を正しくデコードして表示する"""
    # サンプルのバイト文字列（一部）
    sample_bytes = b'\n\n\xe3\x83\x93\xe3\x83\xbc\xe3\x83\x88\xe3\x81\x9f\xe3\x81\x91\xe3\x81\x97\xe3\x81\xae\xe6\x95\x99\xe7\xa7\x91\xe6\x9b\xb8\xe3\x81\xab\xe8\xbc\x89\xe3\x82\x89\xe3\x81\xaa\xe3\x81\x84\xe6\x97\xa5\xe6\x9c\xac\xe4\xba\xba\xe3\x81\xae\xe8\xac\x8e'
    
    # UTF-8でデコード
    decoded_text = sample_bytes.decode('utf-8')
    print("デコード結果:")
    print(decoded_text)

def read_and_decode_file(file_path):
    """ファイルを読み込み、正しくデコードして表示する"""
    try:
        with open(file_path, 'rb') as f:
            # バイナリモードで読み込み
            content = f.read(1000)  # 最初の1000バイトだけ読み込む
            
        # UTF-8でデコード
        decoded_text = content.decode('utf-8')
        print(f"ファイル '{file_path}' の最初の部分:")
        print(decoded_text)
        
        return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("=== バイト文字列デコードのデモ ===")
    decode_bytes_sample()
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\n=== ファイルの読み込みとデコード: {file_path} ===")
        read_and_decode_file(file_path)
    else:
        print("\n使用方法: python decode_bytes.py [ファイルパス]")
        print("ファイルパスを指定すると、そのファイルの内容をデコードして表示します。")