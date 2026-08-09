#!/usr/bin/env python3
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
from typing import Optional, Tuple

import aiohttp
import cv2
import discord
import keyboard
import numpy as np
import requests

def sendCatchCommand(actionType: str, textContent: Optional[str] = None) -> None:
    textMap = {
        "buy": "<@716390085896962058> inc buy 30minutes 30seconds -y",
        "resume": "<@716390085896962058> inc resume",
        "pause": "<@716390085896962058> inc pause",
        "pause_clip": "<@716390085896962058> inc pause",
    }

    rawText = ""
    if actionType in textMap:
        rawText = textMap[actionType]
    elif actionType == "catch" and textContent:
        rawText = f"@Pokétwo#8236 c {textContent}"

    if not rawText:
        return

    escapedText = rawText.replace(" ", "%s").replace("<", "\\<").replace(">", "\\>").replace("&", "\\&")
    
    subprocess.run(
        ["adb", "shell", "input", "text", escapedText],
        check=True
    )
    subprocess.run(
        ["adb", "shell", "input", "keyevent", "66"],
        check=True
    )

def sendCatchCommandWin(actionType: str, textContent: Optional[str] = None):
    if actionType == "buy":
        text = "<@716390085896962058> inc buy 30minutes 30seconds -y"
        subprocess.run(
            ["clip"],
            input=text.strip(),
            encoding="utf-16",
            check=True,
        )
        keyboard.press_and_release("ctrl+v")
        keyboard.press_and_release("enter")

    elif actionType == "resume":
        text = "<@716390085896962058> inc resume"
        subprocess.run(
            ["clip"],
            input=text.strip(),
            encoding="utf-16",
            check=True,
        )
        keyboard.press_and_release("ctrl+v")
        keyboard.press_and_release("enter")

    elif actionType == "pause":
        text = "<@716390085896962058> inc pause"
        subprocess.run(
            ["clip"],
            input=text.strip(),
            encoding="utf-16",
            check=True,
        )
        keyboard.press_and_release("ctrl+v")
        keyboard.press_and_release("enter")

    elif actionType == "pause_clip":
        os.system("echo ^<@716390085896962058^> inc pause | clip")
        keyboard.press_and_release("ctrl+v")
        keyboard.press_and_release("enter")

    elif actionType == "catch":
        if textContent:
            keyboard.write(f"@Pokétwo#8236 c {textContent}")
            keyboard.press_and_release("enter")


def resizeImage(image, maxSize=640):
    h, w = image.shape[:2]
    if max(h, w) <= maxSize:
        return image
    scale = maxSize / max(h, w)
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def buildCombinedIndex(databaseDir=".", outputFile="db_features.pkl"):
    if not os.path.exists(databaseDir):
        print("錯誤：找不到資料夾")
        return

    orb = cv2.ORB_create(nfeatures=500)
    sift = cv2.SIFT_create()

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
        img = cv2.imread(path, 0)
        if img is None:
            continue

        img = resizeImage(img)

        kpOrb, desOrb = orb.detectAndCompute(img, None)
        if desOrb is not None:
            databaseFeatures["orb"][filename] = desOrb

        kpSift, desSift = sift.detectAndCompute(img, None)
        if desSift is not None:
            databaseFeatures["sift"][filename] = desSift

        count += 1
        if count % 100 == 0:
            print(f"已處理 {count} 張...")

    with open(outputFile, "wb") as f:
        pickle.dump(databaseFeatures, f)

    print("--------------------------------")
    print(f"雙重索引建立完成！有效圖片數: {count}")


def extractNumber(text):
    if not text:
        return None
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else None


def loadPokemonMapping(filePath):
    mapping = {}
    encodings = ["utf-8-sig", "utf-8", "cp950", "gbk"]
    for enc in encodings:
        try:
            with open(filePath, "r", encoding=enc) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        cleanId = extractNumber(row[0])
                        if cleanId:
                            mapping[cleanId] = row[2].strip()
            return mapping
        except Exception:
            continue
    return {}


def runAutoCatch():
    if not os.path.exists("failed"):
        os.makedirs("failed")

    failDown = True
    discordBotToken = input("TOKEN: ") or "預設"
    try:
        targetChannelId = int(input("頻道ID: "))
    except Exception:
        targetChannelId = int("預設")

    targetUserId = 716390085896962058
    indexFile = r"db_features.pkl"
    pokeListFile = r"pokelist.csv"

    try:
        yourId = int("YOUR ID")
    except Exception:
        yourId = 0

    maxWorkers = 4
    executor = ThreadPoolExecutor(max_workers=maxWorkers)
    globalData = None
    pokemonNameMap = {}

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("discord_poke_monitor")

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    def identifyProcess(targetImg, mode="ORB"):
        if mode == "ORB":
            detector = cv2.ORB_create(nfeatures=500)
            indexParams = dict(
                algorithm=6,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            searchParams = dict(checks=50)
            ratio = 0.75
            minVotes = 8
        else:
            detector = cv2.SIFT_create()
            indexParams = dict(algorithm=1, trees=5)
            searchParams = dict(checks=50)
            ratio = 0.8
            minVotes = 5

        kpQuery, desQuery = detector.detectAndCompute(targetImg, None)
        if desQuery is None:
            return None

        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        trainDes = globalData[mode.lower()]["descriptors"]
        flann.add([trainDes])
        flann.train()

        matches = flann.knnMatch(desQuery, k=2)
        votes = []
        for mTuple in matches:
            if len(mTuple) == 2:
                m, n = mTuple
                if m.distance < ratio * n.distance:
                    idx = m.trainIdx
                    if idx < len(globalData[mode.lower()]["indices"]):
                        votes.append(globalData[mode.lower()]["indices"][idx])

        if not votes:
            return None
        bestImgId, count = Counter(votes).most_common(1)[0]

        if count >= minVotes:
            return (globalData[mode.lower()]["filenames"][bestImgId], count)
        return None

    def dualIdentifyWorker(data: bytes) -> Optional[Tuple[str, int, str]]:
        try:
            npArr = np.frombuffer(data, np.uint8)
            targetImg = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
            if targetImg is None:
                return None

            h, w = targetImg.shape
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                targetImg = cv2.resize(
                    targetImg, None, fx=scale, fy=scale
                )

            res = identifyProcess(targetImg, mode="ORB")
            if res:
                return (res[0], res[1], "ORB")

            logger.info("ORB 辨識失敗，啟動 SIFT 備援路徑...")
            res = identifyProcess(targetImg, mode="SIFT")
            if res:
                return (res[0], res[1], "SIFT")

            return None
        except Exception as e:
            logger.error(f"辨識錯誤: {e}")
            return None

    @client.event
    async def on_ready():
        nonlocal globalData, pokemonNameMap
        print(f"\n登入身分: {client.user}")

        if not os.path.exists(indexFile):
            print(f"錯誤: 找不到 {indexFile}")
            await client.close()
            return

        with open(indexFile, "rb") as f:
            rawData = pickle.load(f)

        globalData = {"orb": {}, "sift": {}}
        for mode in ["orb", "sift"]:
            allDes, allIdx, allFnames = [], [], []
            dtype = np.uint8 if mode == "orb" else np.float32
            for i, (fn, des) in enumerate(rawData[mode].items()):
                allFnames.append(fn)
                allDes.extend(des)
                allIdx.extend([i] * len(des))
            globalData[mode] = {
                "descriptors": np.array(allDes, dtype=dtype),
                "indices": allIdx,
                "filenames": allFnames,
            }

        pokemonNameMap = loadPokemonMapping(pokeListFile)
        print("系統就緒，已載入雙重特徵庫。")

    @client.event
    async def on_message(message):
        if message.author.id == 874910942490677270:
            return
        done = False
        if message.embeds:
            for embed in message.embeds:
                checkText = ""
                if embed.footer and embed.footer.text:
                    checkText += embed.footer.text
                if embed.description:
                    checkText += embed.description
                for field in embed.fields:
                    checkText += f" {field.value}"

                if "Spawns Remaining: 0" in checkText:
                    done = True
                    if done:
                        await asyncio.sleep(5)
                        sendCatchCommand("buy")
                        await asyncio.sleep(0.2)

        if message.content.lower() == "ir":
            sendCatchCommand("resume")
            await asyncio.sleep(0)
            return

        if (
            message.content.lower() == "ip" and message.author.id == yourId
        ) or "Whoa there. Please tell us you're human!" in message.content:
            sendCatchCommand("pause")
            await asyncio.sleep(0)
            return

        if message.author == client.user:
            return
        if message.channel.id != targetChannelId:
            return
        if message.author.id != targetUserId:
            return

        if re.search(r"https?://\S+", message.content):
            sendCatchCommand("pause_clip")
            await asyncio.sleep(0.2)
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
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            executor, dualIdentifyWorker, imgBytes
                        )

                        if result:
                            matchedFilename, score, engine = result
                            pokeId = extractNumber(matchedFilename)
                            englishName = pokemonNameMap.get(
                                pokeId, "Unknown Name"
                            )
                            print(f"   - 辨識引擎: {engine}")
                            print(
                                f"   - 英文名稱: {englishName} (Score: {score})"
                            )
                            sendCatchCommand("catch", textContent=englishName)
                        else:
                            if failDown:
                                savePath = os.path.join(
                                    "failed",
                                    f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                )
                                with open(savePath, "wb") as f:
                                    f.write(imgBytes)

    client.run(discordBotToken)


def runAutoScan():
    discordBotToken = input("TOKEN：") or "預設TOKEN"
    try:
        targetGuildId = int(input("伺服器ID：")) or "預設ID"
    except Exception:
        print("ID錯誤，請重啟程式。")

    targetUserId = 716390085896962058
    outputCsv = "match_results.csv"
    indexFile = "db_features.pkl"

    maxConcurrentChannels = 15
    maxWorkers = 4

    globalIndexData = None
    executor = ThreadPoolExecutor(max_workers=maxWorkers)
    semaphore = asyncio.Semaphore(maxConcurrentChannels)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("discord_img_matcher")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    client = discord.Client(intents=intents)

    def identifyImageWorker(data: bytes) -> Optional[Tuple[str, int]]:
        if globalIndexData is None:
            return None
        try:
            npArr = np.frombuffer(data, np.uint8)
            targetImg = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
            if targetImg is None:
                return None

            h, w = targetImg.shape
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                targetImg = cv2.resize(
                    targetImg, None, fx=scale, fy=scale
                )

            orb = cv2.ORB_create(nfeatures=500)
            kpQuery, desQuery = orb.detectAndCompute(targetImg, None)
            if desQuery is None:
                return None

            indexParams = dict(
                algorithm=6,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            flann = cv2.FlannBasedMatcher(indexParams, dict(checks=50))
            flann.add([globalIndexData["descriptors"]])
            flann.train()

            matches = flann.knnMatch(desQuery, k=2)
            votes = []
            for mTuple in matches:
                if len(mTuple) == 2:
                    m, n = mTuple
                    if m.distance < 0.75 * n.distance:
                        idx = m.trainIdx
                        if idx < len(globalIndexData["indices"]):
                            votes.append(globalIndexData["indices"][idx])

            if not votes:
                return None
            bestImgId, count = Counter(votes).most_common(1)[0]

            if count >= 8:
                return (globalIndexData["filenames"][bestImgId], count)
            return None
        except Exception as e:
            logger.error(f"辨識過程發生錯誤: {e}")
            return None

    async def processSingleChannel(channel, session, csvWriter, userId):
        async with semaphore:
            try:
                perms = channel.permissions_for(channel.guild.me)
                if not perms.read_message_history:
                    return

                targetMsg = None
                async for msg in channel.history(limit=100):
                    if msg.author.id == userId:
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

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        executor, identifyImageWorker, imgBytes
                    )

                    if result:
                        matchedFile, count = result
                        print(
                            f"[成功] 頻道: {channel.name.ljust(15)} | 匹配: {matchedFile} ({count} 票)"
                        )
                        csvWriter.writerow(
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

            except Exception:
                pass

    @client.event
    async def on_ready():
        nonlocal globalIndexData
        print(f"機器人已連線: {client.user}")

        if not os.path.exists(indexFile):
            print(f"錯誤: 找不到 {indexFile}")
            await client.close()
            return

        print("正在優化索引資料...")
        with open(indexFile, "rb") as f:
            raw = pickle.load(f)

        allDes, allIdx, allFnames = [], [], []
        for i, (fn, des) in enumerate(raw.items()):
            if des is not None:
                allFnames.append(fn)
                allDes.extend(des)
                allIdx.extend([i] * len(des))

        globalIndexData = {
            "descriptors": np.array(allDes, dtype=np.uint8),
            "indices": allIdx,
            "filenames": allFnames,
        }

        guild = client.get_guild(targetGuildId)
        if not guild:
            print("找不到指定伺服器")
            await client.close()
            return

        f = open(outputCsv, "a", newline="", encoding="utf-8-sig")
        writer = csv.writer(f)
        if os.stat(outputCsv).st_size == 0:
            writer.writerow(
                ["頻道", "訊息ID", "發送者", "匹配檔案", "票數", "URL"]
            )

        print(f"開始並行掃描 {guild.name}，請稍候...\n")
        async with aiohttp.ClientSession() as session:
            channels = [
                c
                for c in guild.channels
                if isinstance(c, discord.TextChannel)
            ]
            tasks = [
                processSingleChannel(ch, session, writer, targetUserId)
                for ch in channels
            ]
            await asyncio.gather(*tasks)

        f.close()
        print("\n[完成] 所有頻道已判讀完畢，結果存於 CSV。")
        await client.close()

    client.run(discordBotToken)


def runFailCheck():
    filePath = "failed.txt"

    if not os.path.exists(filePath):
        print(f"找不到檔案 {filePath}，請確定它與本程式放在同一個資料夾。")
        return

    with open(filePath, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = (
        r"(https://cdn\.discordapp\.com/[^\s:]+?(?:hm=[a-f0-9]{64}|size=\d+))"
    )
    urls = re.findall(pattern, content)

    outputDir = "downloaded_failed_attachments"
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

    if args.mode == "catch":
        runAutoCatch()
    elif args.mode == "scan":
        runAutoScan()
    elif args.mode == "failcheck":
        runFailCheck()
    elif args.mode == "buildindex":
        buildCombinedIndex(".")


if __name__ == "__main__":
    main()

