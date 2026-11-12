---
description: 針對要求進行machine learning model 或演算法設計以及開發
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
        其中版本被分為
            ## Promising
                值得進一步探索的架構或方法。

            ## Experimental
                尚未充分測試的架構或方法。

            ## Baseline
                原始參考模型或方法。

            ## 已棄用
                已確認沒有探索價值的架構或方法。
    請先讀取這些檔案來同步進度。若有檔案缺失或與描述不符，請建立或修改這些檔案使其一致以利後續開發。CLAUDE.md與README.md的功能請不要與其他檔案重疊，這兩者應只放置簡要且概念性的說明。演算法/model/code都會在之後大幅度改動，因此千萬不要寫死，例如CLAUDE.md 不應該預先定義：
        固定的 model architecture。
        固定的演算法

3. 請在進行實驗時同步編輯LOG.md，紀錄目前的設計、實驗以及結果，不可改寫已有內容。同時請同步覆寫 SUMMARY.md 的「目前狀態」區塊（≤20 行）。
4. 評估時優先使用少量具有代表性的 cases 進行初步評估。
5. 若有出現可能有價值的新版本，請將其說明加入checkpoints.md，說明請包括結果（各指標變化、優缺點等）利於其他開發者或agent理解以便進行實驗。若某版本有improve請同步更新。若已確定某些版本效果不佳，請將確定棄用的版本移到檔案最下方的「已棄用」區塊，保留簡要說明，不要整段刪除。不要更新baseline，其只有在確認某個方法相較baseline有巨大進步，經使用者同意後才可修改
6. 請盡量避免修改此資料夾外的code。若需要修改，請新增一個 flag（預設關閉）控制是否使用修改後的邏輯；flag 關閉時，行為必須與修改前完全相同，並執行該 sub-project 現有的測試或 smoke test 確認。請同步更新資料夾外目標的LOG.md / SUMMARY.md / checkpoints.md。 