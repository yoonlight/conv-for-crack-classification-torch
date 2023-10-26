# Settings

- 환경: Windows
- 개발 플랫폼: VSCODE

- virtualenv 설치

```sh
python -m venv venv
```

- virtualenv 실행

```sh
.\venv\Scripts\activate
```

- package 설치

```sh
pip install -r requirements.txt
```

- dataset 다운로드
- https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification
- 폴더명 archive로 되어 있는데 crack으로 변경 후 datasets로 이동

- 학습

```sh
python run.py -m shallow -a cpu
python run.py -m lenet -a cpu
python run.py -m alexnet -a cpu
```
