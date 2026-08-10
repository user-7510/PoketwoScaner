# Poketwo Scaner Usage

A multi-functional Python application integrated with a Discord bot. It utilizes OpenCV (ORB and SIFT algorithms) to perform dual-engine image recognition, automated channel monitoring, historical message scanning, and batch downloading of unidentified images.

## Features

The script operates in four primary modes:

* **`buildindex` (Build Feature Database):** Reads images from the `data/` directory, extracts ORB and SIFT features, and compiles them into a serialized database (`db_features.pkl`) for rapid matching.
* **`catch` (Auto-Catch / Monitor):** Listens to a specified Discord channel. When target images are posted, it runs dual-engine recognition against the database. Successful matches trigger an automated action; failed matches have their URLs logged to `failed.txt` and are downloaded locally.
* **`scan` (Channel Scan):** Iterates through historical messages in all text channels of a specific Guild (Server). Identifies images sent by a target user, performs feature matching, and exports the results to `match_results.csv`.
* **`failcheck` (Download Failed Attachments):** Parses `failed.txt` for Discord attachment URLs and batch-downloads them into the `downloaded_failed_attachments/` directory for further manual inspection.

## Prerequisites

* **Python 3.8+**
* Required Python packages:
```bash
pip install discord.py aiohttp opencv-python numpy requests keyboard

```



## Directory Structure

Ensure the following files and directories exist in your working environment before running the tool:

* `main.py`: The main script.
* `data/`: Directory containing the raw images used to build the feature index.
* `pokelist.csv`: A CSV file containing ID to Name mappings (required for `catch` mode). Format: `[ID, ..., Name]`.
* `db_features.pkl`: Generated automatically after running the `buildindex` mode.

## Usage

The big files can be download from https://drive.google.com/drive/folders/1CuRARgTeYvtyszGmSS_4EGpwWi3DLTj4?usp=sharing

Run the script via the command line by specifying the execution mode:
### 0. Clone the reposory

```bash
git clone https://github.com/user-7510/PoketwoScaner
cd PoketwoScaner

```

### 1. Build Index

```bash
python main.py buildindex

```
- You can also directly download the zip from: https://drive.google.com/drive/folders/1CuRARgTeYvtyszGmSS_4EGpwWi3DLTj4?usp=sharing

### 2. Auto-Catch Mode

```bash
python main.py catch

```

*Note: The console will prompt you to enter your Discord Bot `TOKEN` and the target `Channel ID`.*

### 3. Scan Mode

```bash
python main.py scan

```

*Note: The console will prompt you to enter your Discord Bot `TOKEN` and the target `Guild ID`.*

### 4. Download Failed Attachments

```bash
python main.py failcheck

```
