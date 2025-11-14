# PowerShell script để cài đặt tất cả VS Code extensions được khuyến nghị
# Chạy script này: .\scripts\install_vscode_extensions.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing VS Code Extensions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Danh sách extensions
$extensions = @(
    # Python Development
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    
    # Jupyter Notebooks
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-keymap",
    "ms-toolsai.jupyter-renderers",
    
    # Git & Version Control
    "eamodio.gitlens",
    "mhutchie.git-graph",
    "donjayamanne.githistory",
    
    # Code Quality & Linting
    "charliermarsh.ruff",
    "ms-python.ruff",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    
    # Data Science & ML
    "ms-toolsai.vscode-jupyter-cell-tags",
    "ms-toolsai.vscode-jupyter-slideshow",
    
    # Productivity
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.makefile-tools",
    "ms-vscode.powershell",
    
    # Markdown & Documentation
    "yzhang.markdown-all-in-one",
    "davidanson.vscode-markdownlint",
    "bierner.markdown-preview-github-styles",
    
    # Code Navigation & Search
    "alefragnani.project-manager",
    "alefragnani.bookmarks",
    "streetsidesoftware.code-spell-checker",
    
    # Themes & Icons
    "pkief.material-icon-theme",
    "zhuangtongfa.material-theme",
    "github.github-vscode-theme",
    
    # Docker & Containers
    "ms-azuretools.vscode-docker",
    
    # Remote Development
    "ms-vscode-remote.remote-ssh",
    "ms-vscode-remote.remote-containers",
    
    # Testing
    "ms-python.pytest",
    "littlefoxteam.vscode-python-test-adapter",
    
    # Database
    "ms-ossdata.vscode-postgresql",
    "cweijan.vscode-database-client2",
    
    # Performance & Monitoring
    "ms-vscode.vscode-typescript-next",
    "formulahendry.code-runner"
)

# Kiểm tra xem VS Code có được cài đặt không
$codeCommand = Get-Command code -ErrorAction SilentlyContinue
if (-not $codeCommand) {
    Write-Host "ERROR: VS Code 'code' command not found!" -ForegroundColor Red
    Write-Host "Please install VS Code and add it to PATH" -ForegroundColor Yellow
    Write-Host "Or run: Install-Module -Name vscode -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found VS Code installation" -ForegroundColor Green
Write-Host ""

$installed = 0
$failed = 0
$skipped = 0

foreach ($ext in $extensions) {
    Write-Host "Installing: $ext" -ForegroundColor Yellow -NoNewline
    
    # Kiểm tra xem extension đã được cài đặt chưa
    $installedExts = code --list-extensions 2>$null
    if ($installedExts -contains $ext) {
        Write-Host " [SKIPPED - Already installed]" -ForegroundColor Gray
        $skipped++
        continue
    }
    
    # Cài đặt extension
    $result = code --install-extension $ext --force 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " [OK]" -ForegroundColor Green
        $installed++
    }
    else {
        Write-Host " [FAILED]" -ForegroundColor Red
        Write-Host "  Error: $result" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installed: $installed" -ForegroundColor Green
Write-Host "Skipped:  $skipped" -ForegroundColor Gray
Write-Host "Failed:   $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($failed -eq 0) {
    Write-Host "All extensions installed successfully!" -ForegroundColor Green
}
else {
    Write-Host "Some extensions failed to install. Please check errors above." -ForegroundColor Yellow
}


