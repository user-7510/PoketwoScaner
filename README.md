# Pokétwo Tools Suite / Pokétwo 工具套件

[English](#english) | [繁體中文](#繁體中文)

---

<a name="english"></a>
## English

### Overview
This project provides a unified Python tool suite for **Pokétwo** (Discord Bot). It integrates feature database building, automated image scanning/recognition, auto-catching, and failed attachment downloading into a single script (`app.py`).

To facilitate multi-platform development (e.g., Discord, Line, Telegram, or Webhook integration), all keyboard interaction and automated input operations are encapsulated into an independent modular function `sendCatchCommand()`.

---

### Features & Functions

1. **`buildindex`**: Generates a feature database (`db_features.pkl`) using ORB and SIFT algorithms from a local image folder.
2. **`catch`**: Monitors Discord channel messages in real time, recognizes spawned Pokémon images, and sends catch commands automatically.
3. **`scan`**: Scans channels across a Discord server in parallel, matches Pokémon images against the feature database, and outputs results to a CSV file.
4. **`failcheck`**: Reads `failed.txt` containing raw image attachment links and downloads all missing/failed images locally for debugging or retraining.

---

### Prerequisites & Installation

#### Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- Discord.py (`discord.py`)
- Keyboard (`keyboard`)
- Aiohttp (`aiohttp`)
- Requests (`requests`)
- NumPy (`numpy`)

#### Installation
```bash
pip install opencv-python discord.py keyboard aiohttp requests numpy

File Structure
.
├── app.py           # Unified main application script
├── pokelist.csv     # Pokémon ID to English name mapping
├── db_features.pkl  # Generated feature database (after buildindex)
├── failed.txt       # Input file for failcheck mode
└── failed/          # Folder where unrecognized images are stored

Usage
Run app.py with the desired mode argument:
1. Build Feature Database
Constructs ORB and SIFT image feature indexes from images in the current directory:
python app.py buildindex

2. Auto Catch Mode
Monitors a target Discord channel and sends catch commands automatically:
python app.py catch

3. Channel Scan Mode
Scans historical messages in server channels and logs matches to match_results.csv:
python app.py scan

4. Download Failed Attachments
Parses failed.txt for Discord CDN URLs and downloads them locally:
python app.py failcheck

Multi-Platform Customization
All platform-specific input and action logic (such as keyboard key presses or clipboard pasting) is encapsulated inside the sendCatchCommand() function:
def sendCatchCommand(actionType: str, textContent: Optional[str] = None):
    # Customize keybindings, Webhook calls, or API responses for different platforms here
    ...

To adapt this tool for other platforms (e.g., Telegram Bot, Line Bot, or custom REST APIs), simply modify sendCatchCommand() without altering the core image processing or event-driven recognition logic.
<a name="繁體中文"></a>
繁體中文
專案簡介
本專案為 Pokétwo (Discord Bot) 的整合型 Python 自動化工具套件。將原先分散的特徵庫建置、自動辨識抓取、伺服器頻道掃描以及失敗附件下載等功能整合至單一腳本 (app.py) 中。
為了方便跨平台開發（例如：轉移至 Telegram、Line 或 Webhook），所有涉及鍵盤模擬與輸入控制的邏輯皆已獨立封裝至 sendCatchCommand() 函數中。
主要功能模組
 * buildindex (建立特徵庫)：利用 ORB 與 SIFT 雙重演算法，對本地圖像進行特徵提取並生成索引檔 (db_features.pkl)。
 * catch (自動抓取)：即時監聽指定 Discord 頻道的訊息，自動辨識出現的寶可夢影像並傳送抓取指令。
 * scan (頻道掃描)：多線程並行掃描伺服器內所有文字頻道，將辨識比對結果匯出至 CSV 檔案。
 * failcheck (失敗檔下載)：解析 failed.txt 中的 Discord 附件網址，批量下載辨識失敗的圖片以供後續補強。
環境建置與安裝
系統需求
 * Python 3.8 或以上版本
 * OpenCV (opencv-python)
 * Discord.py (discord.py)
 * Keyboard (keyboard)
 * Aiohttp (aiohttp)
 * Requests (requests)
 * NumPy (numpy)
安裝依賴套件
pip install opencv-python discord.py keyboard aiohttp requests numpy

專案檔案結構
.
├── app.py           # 整合主程式腳本
├── pokelist.csv     # 寶可夢編號與英文名稱對照表
├── db_features.pkl  # 特徵庫索引檔 (執行 buildindex 後生成)
├── failed.txt       # 失敗連結清單 (failcheck 模式輸入檔)
└── failed/          # 辨識失敗影像自動存檔目錄

使用說明
執行 app.py 時傳入對應的模式引數：
1. 建立影像特徵庫
掃描當前目錄下的圖片檔並建立 ORB + SIFT 特徵索引：
python app.py buildindex

2. 啟動自動抓取模式
監聽 Discord 頻道並在辨識成功時自動執行抓取操作：
python app.py catch

3. 啟動伺服器頻道掃描
批量檢索伺服器歷史訊息並將比對結果儲存至 match_results.csv：
python app.py scan

4. 下載失敗附件圖片
讀取 failed.txt 內的 CDN 連結並自動下載至指定資料夾：
python app.py failcheck

跨平台二次開發說明
所有與作業系統或平台互動的輸入邏輯（如 keyboard 模擬按鍵與剪貼簿操作）均集中在 sendCatchCommand() 函數：
def sendCatchCommand(actionType: str, textContent: Optional[str] = None):
    # 可在此處替換為其他平台的發送邏輯 (如 Telegram API、Line Bot SDK 或 Webhook)
    ...

若欲改造成其他平台的機器人，只需修改此函數內的動作實作，無需更動底層影像處理與比對的核心邏輯。


