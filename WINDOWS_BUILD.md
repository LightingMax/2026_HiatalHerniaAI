# Windows CPU 打包说明

## 1) 在 Windows 本机打包（推荐）

1. 安装 Python 3.10+（64 位）。
2. 在项目目录打开 `cmd` 或 PowerShell。
3. 执行：
   - `build_windows_cpu.bat`（cmd）
   - 或 `./build_windows_cpu.ps1`（PowerShell）
4. 打包完成后，`exe` 在：`dist/HiatalHerniaAI_CPU.exe`

## 2) 在 Mac 上触发 Windows 自动构建（GitHub Actions）

1. 把项目推送到 GitHub 仓库。
2. 打开仓库 -> `Actions`。
3. 找到工作流：`Build Windows EXE (CPU)`。
4. 点击 `Run workflow`。
5. 运行结束后在 `Artifacts` 下载 `HiatalHerniaAI_CPU_windows`。

## 3) 模型文件

`liekongshanapp.spec` 会自动把下面模型文件一起打包（若文件存在）：
- `best_model_liekongshan.pth`
- `best_model_liekongshan_2classes.pth`

## 4) 注意事项

- 在 macOS 直接运行 `pyinstaller` 只能产出 macOS 可执行文件，不能原生产出 Windows `exe`。
- Windows 首次启动较慢（会先解压运行时环境）。
- 本项目将 `torch==2.8.0`、`torchvision==0.23.0` 固定为 CPU 轮子，减少 Windows DLL 初始化失败概率。
- 如果出现 `invalid load key, 'v'`，通常是打包到了 Git LFS 指针文件（不是实际 `.pth`）。请确保构建前已执行 `git lfs pull`。
- 如果出现 `failed finding central directory`，通常是 `.pth` 文件损坏或传输不完整，请用原始权重重新覆盖后再构建。
