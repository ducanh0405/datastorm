# VS Code Extensions - Developer Setup Guide

Danh sách các VS Code extensions được khuyến nghị cho dự án E-Grocery Forecaster.

## 🚀 Cài đặt nhanh

### Windows (PowerShell)
```powershell
.\scripts\install_vscode_extensions.ps1
```

### Linux/Mac (Bash)
```bash
bash scripts/install_vscode_extensions.sh
```

### Thủ công
Mở VS Code và nhấn `Ctrl+Shift+X` (hoặc `Cmd+Shift+X` trên Mac), sau đó cài đặt từng extension từ danh sách bên dưới.

---

## 📦 Danh sách Extensions

### 🐍 Python Development (Bắt buộc)

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Python** | `ms-python.python` | Hỗ trợ Python với IntelliSense, debugging, testing |
| **Pylance** | `ms-python.vscode-pylance` | Language server nhanh và chính xác cho Python |
| **Debugpy** | `ms-python.debugpy` | Debugger cho Python |
| **Black Formatter** | `ms-python.black-formatter` | Code formatter theo chuẩn Black |
| **isort** | `ms-python.isort` | Sắp xếp imports tự động |
| **Flake8** | `ms-python.flake8` | Linter cho Python |
| **Pylint** | `ms-python.mypy-type-checker` | Type checker cho Python |

### 📓 Jupyter Notebooks (Bắt buộc)

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Jupyter** | `ms-toolsai.jupyter` | Hỗ trợ Jupyter notebooks |
| **Jupyter Keymap** | `ms-toolsai.jupyter-keymap` | Keyboard shortcuts cho Jupyter |
| **Jupyter Renderers** | `ms-toolsai.jupyter-renderers` | Renderers cho các loại output |

### 🔧 Code Quality & Linting

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Ruff** | `charliermarsh.ruff` | Linter nhanh cho Python (thay thế Flake8) |
| **ESLint** | `dbaeumer.vscode-eslint` | Linter cho JavaScript/TypeScript |
| **Prettier** | `esbenp.prettier-vscode` | Code formatter đa ngôn ngữ |

### 🔀 Git & Version Control

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **GitLens** | `eamodio.gitlens` | Tăng cường Git capabilities |
| **Git Graph** | `mhutchie.git-graph` | Visualize Git history |
| **Git History** | `donjayamanne.githistory` | Xem lịch sử Git |

### 📊 Data Science & ML

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Jupyter Cell Tags** | `ms-toolsai.vscode-jupyter-cell-tags` | Quản lý tags cho Jupyter cells |
| **Jupyter Slideshow** | `ms-toolsai.vscode-jupyter-slideshow` | Tạo slideshow từ notebooks |

### ⚡ Productivity

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **JSON** | `ms-vscode.vscode-json` | Hỗ trợ JSON |
| **YAML** | `redhat.vscode-yaml` | Hỗ trợ YAML |
| **Makefile Tools** | `ms-vscode.makefile-tools` | Hỗ trợ Makefiles |
| **PowerShell** | `ms-vscode.powershell` | Hỗ trợ PowerShell |

### 📝 Markdown & Documentation

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Markdown All in One** | `yzhang.markdown-all-in-one` | Công cụ Markdown đầy đủ |
| **Markdown Lint** | `davidanson.vscode-markdownlint` | Linter cho Markdown |
| **Markdown Preview** | `bierner.markdown-preview-github-styles` | Preview với GitHub styles |

### 🔍 Code Navigation & Search

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Project Manager** | `alefragnani.project-manager` | Quản lý nhiều projects |
| **Bookmarks** | `alefragnani.bookmarks` | Đánh dấu code |
| **Code Spell Checker** | `streetsidesoftware.code-spell-checker` | Kiểm tra chính tả |

### 🎨 Themes & Icons

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Material Icon Theme** | `pkief.material-icon-theme` | Icon theme đẹp |
| **Material Theme** | `zhuangtongfa.material-theme` | Color theme |
| **GitHub Theme** | `github.github-vscode-theme` | GitHub official theme |

### 🐳 Docker & Containers

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Docker** | `ms-azuretools.vscode-docker` | Quản lý Docker containers |

### 🌐 Remote Development

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Remote - SSH** | `ms-vscode-remote.remote-ssh` | Làm việc qua SSH |
| **Remote - Containers** | `ms-vscode-remote.remote-containers` | Làm việc trong containers |

### 🧪 Testing

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **Pytest** | `ms-python.pytest` | Hỗ trợ pytest |
| **Python Test Adapter** | `littlefoxteam.vscode-python-test-adapter` | Test explorer cho Python |

### 🗄️ Database

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **PostgreSQL** | `ms-ossdata.vscode-postgresql` | Hỗ trợ PostgreSQL |
| **Database Client** | `cweijan.vscode-database-client2` | Client cho nhiều databases |

### ⚙️ Performance & Monitoring

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **TypeScript Next** | `ms-vscode.vscode-typescript-next` | TypeScript support |
| **Code Runner** | `formulahendry.code-runner` | Chạy code snippets |

### 🤖 AI & Copilot (Optional)

| Extension | ID | Mô tả |
|-----------|-----|-------|
| **GitHub Copilot** | `github.copilot` | AI pair programmer |
| **GitHub Copilot Chat** | `github.copilot-chat` | AI chat assistant |

---

## ⚙️ Cấu hình VS Code

Sau khi cài đặt extensions, tạo file `.vscode/settings.json` với cấu hình tối ưu:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[markdown]": {
    "editor.defaultFormatter": "yzhang.markdown-all-in-one"
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true
  },
  "jupyter.askForKernelRestart": false,
  "jupyter.interactiveWindowMode": "perFile"
}
```

---

## 📋 Checklist

- [ ] Cài đặt tất cả Python extensions
- [ ] Cài đặt Jupyter extensions
- [ ] Cài đặt Git extensions
- [ ] Cài đặt Code Quality extensions
- [ ] Cài đặt Productivity extensions
- [ ] Cấu hình VS Code settings
- [ ] Restart VS Code

---

## 🔗 Tài liệu tham khảo

- [VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [GitLens Documentation](https://gitlens.amod.io/)

---

**Lưu ý**: Một số extensions có thể yêu cầu license (như GitHub Copilot). Hãy kiểm tra trước khi cài đặt.


