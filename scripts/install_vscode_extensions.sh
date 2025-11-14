#!/bin/bash
# Bash script để cài đặt tất cả VS Code extensions được khuyến nghị
# Chạy script này: bash scripts/install_vscode_extensions.sh

echo "========================================"
echo "Installing VS Code Extensions"
echo "========================================"
echo ""

# Danh sách extensions
extensions=(
    # Python Development
    "ms-python.python"
    "ms-python.vscode-pylance"
    "ms-python.debugpy"
    "ms-python.black-formatter"
    "ms-python.isort"
    "ms-python.flake8"
    "ms-python.mypy-type-checker"
    
    # Jupyter Notebooks
    "ms-toolsai.jupyter"
    "ms-toolsai.jupyter-keymap"
    "ms-toolsai.jupyter-renderers"
    
    # Git & Version Control
    "eamodio.gitlens"
    "mhutchie.git-graph"
    "donjayamanne.githistory"
    
    # Code Quality & Linting
    "charliermarsh.ruff"
    "ms-python.ruff"
    "dbaeumer.vscode-eslint"
    "esbenp.prettier-vscode"
    
    # Data Science & ML
    "ms-toolsai.vscode-jupyter-cell-tags"
    "ms-toolsai.vscode-jupyter-slideshow"
    
    # Productivity
    "ms-vscode.vscode-json"
    "redhat.vscode-yaml"
    "ms-vscode.makefile-tools"
    "ms-vscode.powershell"
    
    # Markdown & Documentation
    "yzhang.markdown-all-in-one"
    "davidanson.vscode-markdownlint"
    "bierner.markdown-preview-github-styles"
    
    # Code Navigation & Search
    "alefragnani.project-manager"
    "alefragnani.bookmarks"
    "streetsidesoftware.code-spell-checker"
    
    # Themes & Icons
    "pkief.material-icon-theme"
    "zhuangtongfa.material-theme"
    "github.github-vscode-theme"
    
    # Docker & Containers
    "ms-azuretools.vscode-docker"
    
    # Remote Development
    "ms-vscode-remote.remote-ssh"
    "ms-vscode-remote.remote-containers"
    
    # Testing
    "ms-python.pytest"
    "littlefoxteam.vscode-python-test-adapter"
    
    # Database
    "ms-ossdata.vscode-postgresql"
    "cweijan.vscode-database-client2"
    
    # Performance & Monitoring
    "ms-vscode.vscode-typescript-next"
    "formulahendry.code-runner"
)

# Kiểm tra xem VS Code có được cài đặt không
if ! command -v code &> /dev/null; then
    echo "ERROR: VS Code 'code' command not found!"
    echo "Please install VS Code and add it to PATH"
    exit 1
fi

echo "Found VS Code installation"
echo ""

installed=0
failed=0
skipped=0

for ext in "${extensions[@]}"; do
    echo -n "Installing: $ext"
    
    # Kiểm tra xem extension đã được cài đặt chưa
    if code --list-extensions 2>/dev/null | grep -q "^${ext}$"; then
        echo " [SKIPPED - Already installed]"
        ((skipped++))
        continue
    fi
    
    # Cài đặt extension
    if code --install-extension "$ext" --force > /dev/null 2>&1; then
        echo " [OK]"
        ((installed++))
    else
        echo " [FAILED]"
        ((failed++))
    fi
done

echo ""
echo "========================================"
echo "Installation Summary"
echo "========================================"
echo "Installed: $installed"
echo "Skipped:  $skipped"
echo "Failed:   $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo "All extensions installed successfully!"
else
    echo "Some extensions failed to install. Please check errors above."
fi


