# PoketwoScaner

## Download from:
- https://drive.google.com/drive/folders/1--LuI1oT_Q4LjSZK8eaHN5RYPLCoujgE?usp=drive_link

## Steps:
- Download the pokemon images from [Google Drive](https://drive.google.com/drive/folders/1--LuI1oT_Q4LjSZK8eaHN5RYPLCoujgE?usp=drive_link) and put them into "./data/images/ dir".
- Download pokelist.csv from [GitHub](https://github.com/user-7510/PoketwoScaner/blob/main/pokelist.csv) and put it into "./".
- Get your Discord Bot TOKEN and invite it into your guild.
- Get your Discord Guild ID.
- You can also download PoketwoScanner.zip, all of the programs are there.

## Usage:
1. ./pokelistGenerator.py for init
2. ./autoScan.py for just scan or ./autoCatch.py for auto catch
3. ./failCheck.py for chek failed pokemons

## Dependency chain for heritages:
```[1]premodify.py ─┐
[1-2]pkl.py ──────┴─→ db_features.pkl ─┬─→ [2]scanner.py ──→ match_results.csv ─┐
                                        │                                        ├─→ [3]poke-trans.py → output_result.csv
                       pokelist.csv ────┤                                        ┘
                                        ├─→ [2-2]scan_catch.py
                                        ├─→ [2-3]superspamer.py
                                        ├─→ [2-4]scan-catch-pause.py ──→ failed.txt ──→ [3-2]faildown.py
                                        ├─→ [2-5]scan-catch-autoincbuy(.py/-fixed.py)
                                        └─→ catch-final.py / [2-7]catch-final.py
```
