# colony_analysis

Sample data is [here](https://drive.google.com/drive/folders/1oGhDGgz8nyfXi1upOZf3obbIuDiuBR4O?usp=sharing).
## Usage
### To get csv from colony images
#### pict2colony.py
  ```sh
  python pict2colony.py -c 3 -i plates -o pict2colony.csv
  ```
- -c: cut-level
- -i: input csv
- -o: output csv

#### colony2growth.py
  ```sh
  python colony2growth.py -i pict2colony.csv -o colony2growth.csv
  ```
- -i: input csv
- -o: output csv

#### growth2ngwoth.py
  ```sh
  python growth2ngrowth.py -i colony2growth.csv -o growth2ngrowth.csv
  ```
- -i: input csv
- -o: output csv
