#!bin/bash
curl -LJ "https://github.com/Hironsan/anago/raw/master/data/conll2003/en/ner/train.txt" -o "train.txt"
curl -LJ "https://github.com/Hironsan/anago/raw/master/data/conll2003/en/ner/valid.txt" -o "valid.txt"
curl -LJ "https://github.com/Hironsan/anago/raw/master/data/conll2003/en/ner/test.txt" -o "test.txt"