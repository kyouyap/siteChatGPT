# Python Dev Project

## プロジェクト概要

このプロジェクトはPython開発環境のテンプレートです。Docker, Docker Compose, Poetry, そしてVS CodeのDev Containerを使用しています。

## ディレクトリ構造

```
.
├── .devcontainer
│   └── devcontainer.json
├── README.md
├── app
│   ├── Dockerfile
│   ├── inputs
│   ├── notebook
│   ├── outputs
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── src
│   │   └── __init__.py
│   └── test
│       └── __init__.py
└── compose.yaml
```

## セットアップ

### Dev Containerを使用する場合

1. [VS Code](https://code.visualstudio.com/)と[Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)拡張機能をインストールしてください。
2. プロジェクトのルートディレクトリをVS Codeで開きます。
3. 左下の緑色のアイコンをクリックして、`Reopen in Container`を選択します。

これで、VS CodeがDev Container内で開き、すべての依存関係と設定が自動的にインストールされます。

## compose.yaml

Docker Composeの設定は`compose.yaml`にあります。

## Dockerfile

Dockerfileは`app/Dockerfile`にあり、Python 3.9.6をベースにしています。

## pyproject.toml

Poetryの設定は`pyproject.toml`にあります。

## devcontainer.json

VS CodeのDev Container設定は`.devcontainer/devcontainer.json`にあります。

## 開発環境

- Python: 3.9.6
- Poetry: 最新版
- Docker: 最新版

## 依存関係

- `pyproject.toml`を参照

## 開発ツール

- Mypy
- Black
- Pytest
- Pytest-cov
- Poethepoet

## VS Code 拡張機能

- ms-python.python
- GitHub.copilot
- GitHub.copilot-chat
- VisualStudioExptTeam.vscodeintellicode
- Meezilla.json
- ms-azuretools.vscode-docker
- ms-python.autopep8
- ms-python.pylint
- matangover.mypy
- ms-python.black-formatter

## テスト

テストはPytestを使用しています。Dev Container内で以下のコマンドでテストを実行できます。

```bash
pytest -v --cov=. --cov-branch
```

## ライセンス

このプロジェクトはMITライセンスのもとで公開されています。

---

以上がREADMEのテンプレートです。プロジェクトの詳細に応じて適宜修正してください。