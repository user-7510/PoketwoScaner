#!/usrPlugin/env python3
import argparse
import asyncio
import csv
from datetime import datetime
import logging
import os
import pickle
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any, List

import aiohttp
import cv2
import discord
import keyboard
import numpy as np
import requests

logger = logging.getLogger("poke_system")


class ImageUtils:
    @staticmethod
    def resizeImage(image: np.ndarray, maxSize: int = 640) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= maxSize:
            return image
        scale = maxSize / max(h, w)
        return cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def extractNumber(text: Any) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"(\d+)", str(text))
        return int(match.group(1)) if match else None


class FeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.sift = cv2.SIFT_create()
        self.bfOrb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        indexParams = dict(algorithm=1, trees=5)
        searchParams = dict(checks=50)
        self.flannSift = cv2.FlannBasedMatcher(indexParams, searchParams)

        self.globalData: Dict[str, Any] = {}
        self.siftIndexed: bool = False
        self.maxRowsPerChunk: int = 200000

    def buildCombinedIndex(
        self, databaseDir: str = ".", outputFile: str = "db_features.pkl"
    ):
        if not os.path.exists(databaseDir):
            print("錯誤：找不到資料夾")
            return

        databaseFeatures = {"orb": {}, "sift": {}}
        files = [
            f
            for f in os.listdir(databaseDir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"正在建立 {len(files)} 張圖片的雙重索引 (ORB + SIFT)...")

        count = 0
        for filename in files:
            path = os.path.join(databaseDir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = ImageUtils.resizeImage(img)

            _, desOrb = self.orb.detectAndCompute(img, None)
            if desOrb is not None:
                databaseFeatures["orb"][filename] = desOrb

            _, desSift = self.sift.detectAndCompute(img, None)
            if desSift is not None:
                databaseFeatures["sift"][filename] = desSift

            count += 1
            if count % 100 == 0:
                print(f"已處理 {count} 張...")

        with open(outputFile, "wb") as f:
            pickle.dump(databaseFeatures, f)

        print("--------------------------------")
        print(f"雙重索引建立完成！有效圖片數: {count}")

    def loadAndPrepareIndex(self, indexFilePath: str):
        if not os.path.exists(indexFilePath):
            raise FileNotFoundError(f"找不到特徵庫檔案: {indexFilePath}")

        with open(indexFilePath, "rb") as f:
            rawData = pickle.load(f)

        processedData = {}
        for mode in ["orb", "sift"]:
            if mode not in rawData:
                continue
            allDes = []
            allIdx = []
            allFnames = []
            for i, (fn, des) in enumerate(rawData[mode].items()):
                if des is None or len(des) == 0:
                    continue
                allFnames.append(fn)
                allDes.append(des)
                allIdx.extend([i] * len(des))

            if allDes:
                dtype = np.uint8 if mode == "orb" else np.float32
                stackedDes = np.vstack(allDes).astype(dtype)
                indicesArr = np.array(allIdx, dtype=np.int32)

                chunks = []
                indicesChunks = []
                totalRows = stackedDes.shape[0]
                for start in range(0, totalRows, self.maxRowsPerChunk):
                    end = min(start + self.maxRowsPerChunk, totalRows)
                    chunks.append(stackedDes[start:end])
                    indicesChunks.append(indicesArr[start:end])

                processedData[mode] = {
                    "chunks": chunks,
                    "indicesChunks": indicesChunks,
                    "filenames": allFnames,
                }

        self.globalData = processedData

        if "sift" in self.globalData and "chunks" in self.globalData["sift"]:
            self.flannSift.clear()
            for chunk in self.globalData["sift"]["chunks"]:
                self.flannSift.add([chunk.astype(np.float32)])
            self.flannSift.train()
            self.siftIndexed = True

    def identifyProcess(
        self, targetImg: np.ndarray, mode: str = "ORB"
    ) -> Optional[Tuple[str, int]]:
        modeKey = mode.lower()
        if modeKey not in self.globalData or "chunks" not in self.globalData[modeKey]:
            return None

        votes = []
        if mode == "ORB":
            _, desQuery = self.orb.detectAndCompute(targetImg, None)
            if desQuery is None:
                return None
            ratio = 0.75
            minVotes = 8

            chunks = self.globalData[modeKey]["chunks"]
            indicesChunks = self.globalData[modeKey]["indicesChunks"]
            for chunk, idxChunk in zip(chunks, indicesChunks):
                matches = self.bfOrb.knnMatch(desQuery.astype(np.uint8), chunk, k=2)
                for matchPair in matches:
                    if len(matchPair) == 2:
                        m, n = matchPair
                        if m.distance < ratio * n.distance:
                            if m.trainIdx < len(idxChunk):
                                votes.append(idxChunk[m.trainIdx])
        else:
            if not self.siftIndexed:
                return None
            _, desQuery = self.sift.detectAndCompute(targetImg, None)
            if desQuery is None:
                return None
            ratio = 0.8
            minVotes = 5

            matches = self.flannSift.knnMatch(desQuery.astype(np.float32), k=2)
            indicesChunks = self.globalData[modeKey]["indicesChunks"]
            for matchPair in matches:
                if len(matchPair) == 2:
                    m, n = matchPair
                    if m.distance < ratio * n.distance:
                        imgIdx = m.imgIdx
                        trainIdx = m.trainIdx
                        if imgIdx < len(indicesChunks) and trainIdx < len(
                            indicesChunks[imgIdx]
                        ):
                            votes.append(indicesChunks[imgIdx][trainIdx])

        if not votes:
            return None

        bestImgId, count = Counter(votes).most_common(1)[0]
        if count >= minVotes:
            return (self.globalData[modeKey]["filenames"][bestImgId], count)
        return None

    def dualIdentifyWorker(self, data: bytes) -> Optional[Tuple[str, int, str]]:
        try:
            npArr = np.frombuffer(data, np.uint8)
            targetImg = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
            if targetImg is None:
                return None

            targetImg = ImageUtils.resizeImage(targetImg, 640)

            res = self.identifyProcess(targetImg, mode="ORB")
            if res:
                return (res[0], res[1], "ORB")

            logger.info("ORB 辨識失敗，啟動 SIFT 備援路徑...")
            res = self.identifyProcess(targetImg, mode="SIFT")
            if res:
                return (res[0], res[1], "SIFT")

            return None
        except Exception as e:
            logger.error(f"辨識錯誤: {e}")
            return None


class PokemonDataLoader:
    @staticmethod
    def loadPokemonMapping(filePath: str) -> Dict[int, str]:
        mapping = {}
        encodings = ["utf-8-sig", "utf-8", "cp950", "gbk"]
        for enc in encodings:
            try:
                with open(filePath, "r", encoding=enc) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            cleanId = ImageUtils.extractNumber(row[0])
                            if cleanId is not None:
                                mapping[cleanId] = row[2].strip()
                return mapping
            except Exception:
                continue
        return {}


class AutoCatchBot:
    def __init__(
        self,
        token: str,
        targetChannelId: int,
        matcher: FeatureMatcher,
        pokemonMap: Dict[int, str],
        executor: ThreadPoolExecutor,
    ):
        self.token = token
        self.targetChannelId = targetChannelId
        self.matcher = matcher
        self.pokemonMap = pokemonMap
        self.executor = executor
        self.targetUserId = 716390085896962058
        self.failDown = True

        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self.registerEvents()

    def registerEvents(self):
        @self.client.event
        async def on_ready():
            print(f"\n登入身分: {self.client.user}")
            print("自動監聽模組就緒。")

        @self.client.event
        async def on_message(message):
            if message.author.id == 874910942490677270:
                return

            if message.author == self.client.user:
                return
            if message.channel.id != self.targetChannelId:
                return
            if message.author.id != self.targetUserId:
                return

            imageUrl = None
            if message.attachments:
                for att in message.attachments:
                    if any(
                        att.filename.lower().endswith(ext)
                        for ext in [".png", ".jpg", ".jpeg", ".webp"]
                    ):
                        imageUrl = att.url
                        break

            if not imageUrl and message.embeds:
                for embed in message.embeds:
                    if embed.image:
                        imageUrl = embed.image.url
                    elif embed.thumbnail:
                        imageUrl = embed.thumbnail.url
                    if imageUrl:
                        break

            if imageUrl:
                async with aiohttp.ClientSession() as session:
                    async with session.get(imageUrl) as resp:
                        if resp.status == 200:
                            imgBytes = await resp.read()
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                self.executor,
                                self.matcher.dualIdentifyWorker,
                                imgBytes,
                            )

                            if result:
                                matchedFilename, score, engine = result
                                pokeId = ImageUtils.extractNumber(matchedFilename)
                                englishName = self.pokemonMap.get(
                                    pokeId, "Unknown Name"
                                )
                                print("--------------------------------")
                                print(f"[辨識成功] 名稱: {englishName}")
                                print(f" - 匹配檔名: {matchedFilename}")
                                print(f" - 辨識引擎: {engine}")
                                print(f" - 特徵特數: {score}")
                                print("--------------------------------")
                                doAutoCatch(englishName)
                            else:
                                print("[辨識失敗] 無法匹配此圖片。")
                                if self.failDown:
                                    if not os.path.exists("failed"):
                                        os.makedirs("failed")
                                    savePath = os.path.join(
                                        "failed",
                                        f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    )
                                    with open(savePath, "wb") as f:
                                        f.write(imgBytes)

    def start(self):
        self.client.run(self.token)


class AutoScanBot:
    def __init__(
        self,
        token: str,
        targetGuildId: int,
        matcher: FeatureMatcher,
        executor: ThreadPoolExecutor,
    ):
        self.token = token
        self.targetGuildId = targetGuildId
        self.matcher = matcher
        self.executor = executor
        self.targetUserId = 716390085896962058
        self.outputCsv = "match_results.csv"

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        self.client = discord.Client(intents=intents)
        self.registerEvents()

    def identifyImageWorker(self, data: bytes) -> Optional[Tuple[str, int]]:
        try:
            npArr = np.frombuffer(data, np.uint8)
            targetImg = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
            if targetImg is None:
                return None

            targetImg = ImageUtils.resizeImage(targetImg, 640)
            return self.matcher.identifyProcess(targetImg, mode="ORB")
        except Exception as e:
            logger.error(f"辨識過程發生錯誤: {e}")
            return None

    def registerEvents(self):
        @self.client.event
        async def on_ready():
            print(f"機器人已連線: {self.client.user}")

            guild = self.client.get_guild(self.targetGuildId)
            if not guild:
                print("找不到指定伺服器")
                await self.client.close()
                return

            semaphore = asyncio.Semaphore(15)
            fileLock = asyncio.Lock()

            csvFile = open(self.outputCsv, "a", newline="", encoding="utf-8-sig")
            writer = csv.writer(csvFile)
            if os.stat(self.outputCsv).st_size == 0:
                writer.writerow(
                    ["頻道", "訊息ID", "發送者", "匹配檔案", "票數", "URL"]
                )

            print(f"開始並行掃描 {guild.name}，請稍候...\n")

            async def processSingleChannel(channel, session):
                async with semaphore:
                    try:
                        perms = channel.permissions_for(channel.guild.me)
                        if not perms.read_message_history:
                            return

                        targetMsg = None
                        async for msg in channel.history(limit=100):
                            if msg.author.id == self.targetUserId:
                                targetMsg = msg
                                break

                        if not targetMsg:
                            return

                        urls = []
                        for att in targetMsg.attachments:
                            if any(
                                att.filename.lower().endswith(ext)
                                for ext in [".png", ".jpg", ".jpeg", ".webp"]
                            ):
                                urls.append({"url": att.url, "name": att.filename})

                        for emb in targetMsg.embeds:
                            if emb.image:
                                urls.append(
                                    {"url": emb.image.url, "name": "embed_img.png"}
                                )

                        if not urls:
                            return

                        for item in urls:
                            async with session.get(item["url"]) as resp:
                                if resp.status != 200:
                                    continue
                                imgBytes = await resp.read()

                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                self.executor, self.identifyImageWorker, imgBytes
                            )

                            if result:
                                matchedFile, count = result
                                print(
                                    f"[成功] 頻道: {channel.name.ljust(15)} | 匹配: {matchedFile} ({count} 票)"
                                )
                                async with fileLock:
                                    writer.writerow(
                                        [
                                            channel.name,
                                            targetMsg.id,
                                            targetMsg.author,
                                            matchedFile,
                                            count,
                                            item["url"],
                                        ]
                                    )
                            else:
                                print(
                                    f"[失敗] 頻道: {channel.name.ljust(15)} | 無匹配結果"
                                )

                    except Exception as e:
                        logger.debug(f"頻道處理異常 ({channel.name}): {e}")

            async with aiohttp.ClientSession() as session:
                channels = [
                    c
                    for c in guild.channels
                    if isinstance(c, discord.TextChannel)
                ]
                tasks = [processSingleChannel(ch, session) for ch in channels]
                await asyncio.gather(*tasks)

            csvFile.close()
            print("\n[完成] 所有頻道已判讀完畢，結果存於 CSV。")
            await self.client.close()

    def start(self):
        self.client.run(self.token)


class DownloaderService:
    @staticmethod
    def runFailCheck(
        filePath: str = "failed.txt", outputDir: str = "downloaded_failed_attachments"
    ):
        if not os.path.exists(filePath):
            print(f"找不到檔案 {filePath}，請確定它與本程式放在同一個資料夾。")
            return

        with open(filePath, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = (
            r"(https://cdn\.discordapp\.com/[^\s:]+?(?:hm=[a-f0-9]{64}|size=\d+))"
        )
        urls = re.findall(pattern, content)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        print(
            f"解析完畢！共找到 {len(urls)} 個有效的下載連結，準備開始下載...\n"
        )

        successCount = 0
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                filename = url.split("/")[-1].split("?")[0]
                if not filename:
                    filename = f"attachment_{i}.png"

                name, ext = os.path.splitext(filename)
                saveName = f"{name}_{i + 1:03d}{ext}"
                savePath = os.path.join(outputDir, saveName)

                with open(savePath, "wb") as imgFile:
                    imgFile.write(response.content)

                print(f"[{i + 1}/{len(urls)}] 成功下載 -> {saveName}")
                successCount += 1

            except requests.exceptions.RequestException as e:
                print(f"[{i + 1}/{len(urls)}] 下載失敗: {url}\n   └─ 錯誤原因: {e}")
            except Exception as e:
                print(f"[{i + 1}/{len(urls)}] 發生未預期的錯誤: {e}")

        print(f"\n下載作業結束！成功下載了 {successCount} 個檔案。")


class AppRunner:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.matcher = FeatureMatcher()

    def runCatch(self):
        rawToken = input("TOKEN: ").strip()
        discordBotToken = rawToken if rawToken else "DEFAULT_TOKEN"

        rawChannel = input("頻道ID: ").strip()
        try:
            targetChannelId = int(rawChannel)
        except ValueError:
            print("錯誤：輸入的頻道 ID 非有效數字，程式終止。")
            return

        indexFile = "db_features.pkl"
        pokeListFile = "pokelist.csv"

        try:
            self.matcher.loadAndPrepareIndex(indexFile)
        except Exception as e:
            print(f"初始化特徵資料庫失敗: {e}")
            return

        pokemonMap = PokemonDataLoader.loadPokemonMapping(pokeListFile)
        bot = AutoCatchBot(
            token=discordBotToken,
            targetChannelId=targetChannelId,
            matcher=self.matcher,
            pokemonMap=pokemonMap,
            executor=self.executor,
        )
        bot.start()

    def runScan(self):
        rawToken = input("TOKEN：").strip()
        discordBotToken = rawToken if rawToken else "DEFAULT_TOKEN"

        rawGuild = input("伺服器ID：").strip()
        try:
            targetGuildId = int(rawGuild)
        except ValueError:
            print("錯誤：伺服器 ID 格式不正確，程式終止。")
            return

        indexFile = "db_features.pkl"

        try:
            self.matcher.loadAndPrepareIndex(indexFile)
        except Exception as e:
            print(f"初始化特徵資料庫失敗: {e}")
            return

        bot = AutoScanBot(
            token=discordBotToken,
            targetGuildId=targetGuildId,
            matcher=self.matcher,
            executor=self.executor,
        )
        bot.start()

    def runFailCheck(self):
        DownloaderService.runFailCheck()

    def runBuildIndex(self):
        self.matcher.buildCombinedIndex("data")

def doAutoCatch(Name):
    ...

def main():
    parser = argparse.ArgumentParser(
        description="整合自動抓取、頻道掃描與失敗檔下載工具"
    )
    parser.add_argument(
        "mode",
        choices=["catch", "scan", "failcheck", "buildindex"],
        help="執行模式：catch (自動抓取), scan (頻道掃描), failcheck (失敗下載), buildindex (建立特徵庫)",
    )
    args = parser.parse_args()

    runner = AppRunner()
    if args.mode == "catch":
        runner.runCatch()
    elif args.mode == "scan":
        runner.runScan()
    elif args.mode == "failcheck":
        runner.runFailCheck()
    elif args.mode == "buildindex":
        runner.runBuildIndex()


if __name__ == "__main__":
    main()

