# 目標
請使用現有的package優化生成初始mask的流程。

# 說明
本project希望實現3D頭髮模型的自動挑染，並能實現chunky highlighting / money piece highlighting 等不同的挑染效果。為了實現這個目標，我們需要生成初始mask，這個mask將用於後續的挑染操作。

由於長髮的互相遮蓋問題，如chunky highlighting等挑染需要考慮到頭髮的層次和遮擋關係，無法使用簡單的2D mask生成方法。
目前發現，若將交叉數最小化，如使用@../update_contour的FM演算法後，生成的mask會更符合頭髮的層次和遮擋關係。 同時我們也發現，若生成頭髮的邊界是沿著髮流vector field方向或貼著生長點的話交叉數也會比較低。

請設計並實作能夠生成chunky與money piece initial mask 的演算法，希望盡量符合下列特性
- chunky: 挑染髮色呈放射狀，類似西瓜
- money piece: 挑染髮色位於前額


可能有效的方法是使用../flow_field的function先建構髮流向量場與計算div，其中div較高的點幾乎就是生長點。接著盡量以生長點為起點沿著vector field為頭髮分界。可能需要使用BFS之類的方式調整邊界使其完整。請搜尋近期幾何或圖學論文找尋是否有類似問題的更好解法，並自由修改演算法。

請保留設計結構彈性，未來可能需要實做新的挑染風格。

請使用根目錄中 update_contour / contour_algo_test 兩個project來做測試。若想實做某修功能請先參考generate_from_3D_models的md，若已有實做請盡量復用，不要大量實作重複內容避免版本混亂。

實做時請創建檔案mask_gen__chunky__money_piece.log詳細紀錄，並創建Algorithm md file詳細說明最後的最佳演算法。也請創建claude md file.