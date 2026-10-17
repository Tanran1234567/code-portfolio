## 工具：JSON 转 PDF

### 1) 安装依赖

在仓库根目录执行：

```bash
python -m pip install -r backend/requirements.txt
python -m pip install -r tools/requirements.txt
```

### 2) 生成 PDF

把前端导出的 `diplomat_result_*.json` 转为 PDF：

```bash
python tools/diplomat_result_to_pdf.py --input "diplomat_result_demo-001_2026-01-18T02-34-03-015Z.json" --output "diplomat_result_demo-001.pdf"
```

### 3) 中文字体说明

脚本会在 Windows 下自动尝试使用以下字体（按优先级）：

- `C:\Windows\Fonts\msyh.ttc`（微软雅黑）
- `C:\Windows\Fonts\simsun.ttc`（宋体）
- 以及同目录下的 `tools/font.ttf` 或 `tools/font.otf`（你也可以手动放一个字体文件）

如果都找不到，会回退到 `Helvetica`，此时中文可能显示为乱码方块。

