---
description: 請針對要求進行machine learning model 或演算法設計以及開發
argument-hint: <任務說明，可多行>
---
$ARGUMENTS

規則：

1. 開發中的實驗目錄除了 README.md, CLAUDE.md外，還會有以下檔案
    - LOG.md
        實驗的紀錄，供user及其他agent追蹤
    - SUMMARY.md
        重要的實驗結果以及總結，其中有「目前狀態」區塊，描述當前進行的實驗
    - checkpoints.md
        目前可用版本清單，供其他 session 選版本做實驗

2. 設計演算法、模型架構前，先上網尋找近期related work參考，並以使用者輸入為主進行設計。
3. 請創建LOG.md，並在進行實驗時同步編輯，紀錄目前的設計、實驗以及結果，不可改寫已有內容。同時請同步覆寫 SUMMARY.md 的「目前狀態」區塊（≤20 行）。
4. 請創建checkpoints.md，若有出現可能有價值的新版本，請將其說明加入checkpoints.md，說明請包括結果（各指標變化、優缺點等）利於其他開發者或agent理解以便進行實驗。若某版本有improve請同步更新。若已確定某些版本效果不佳，請將確定棄用的版本移到檔案最下方的「已棄用」區塊，保留簡要說明，不要整段刪除。