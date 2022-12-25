# A simple paper-crawler from paperswithcode

```
python run.py --h

usage: run.py [-h] --paper PAPER --result RESULT
              [--max_paper_result MAX_PAPER_RESULT]
              [--max_repository_result MAX_REPOSITORY_RESULT]
              [--verbose | --no-verbose]

optional arguments:
  -h, --help            show this help message and exit
  --paper PAPER         Paper name (accurately) (default: None)
  --result RESULT       File to store the results (default: None)
  --max_paper_result MAX_PAPER_RESULT
                        Max results paper (default: 10)
  --max_repository_result MAX_REPOSITORY_RESULT
                        Max repositories per paper (default: 10)
  --verbose, --no-verbose
                        Show progress (default: True)
```