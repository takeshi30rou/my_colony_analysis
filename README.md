# colony_analysis

## 環境構築

1. [Dockerのインストール](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

2. Docker イメージのビルド

Dockerfileを空のディレクトリンに入れる
```sh
cd <directory for Dockerfile>
docker build -t <IMAGE_NAME> .
```
3. Docker コンテナ起動
docker run --name <CONTAINER_NAME> -it <IMAGE_NAME> bash

## 使い方

- リストの出力を得る場合（他のモジュールとの結合用）
  test_get_table.pyを参照

- CSVの出力を得る場合
  - pict2colony.py
  ```sh
  python pict2colony.py -c 3 -i 5478 -o hoge.csv
  ```
    - -c: cut-level
    - -i: input csv
    - -o: output csv
  
  - colony2growth.py
  ```sh
  python colony2growth.py -i hoge.csv -o hoge2.csv
  ```
    - -i: input csv
    - -o: output csv

  - growth2ngwoth.py
  ```sh
  python growth2ngrowth.py -i hoge2.csv -o hoge3.csv
  ```
    - -i: input csv
    - -o: output csv
