from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="MEXGVFZRD3",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}},
)

query = "休眠模式是什麼"

ans = retriever.invoke(query)
print(ans)

'''

[Document(page_content='準備上路  29   
3.1 機械式鑰匙⾞種  29    
3.1.1 開啟系統電源並解鎖龍頭  29   
3.1.2 關閉系統電源並上鎖龍頭  29   
3.1.3 雙重防盜鎖  30   
3.1.4 開啟座墊下置物箱  31   
3.1.5 休眠模式  32    
3.2 無線鑰匙⾞種  33   
3.2.1 iQ System® 無線智慧鑰匙  34     
3.2.1.1 開啟系統電源並解鎖龍頭  34   3.2.1.2 關閉系統電源及上鎖龍頭  34   
3.2.1.3 開啟座墊下置物箱  34     3.2.2 iQ System® 智慧鑰匙卡  35   
3.2.2.1 iQ System® 智慧鑰匙卡感應器位置  36
3.2.2.2 開啟系統電源並解鎖龍頭  37  
3.2.2.3 關閉系統電源及上鎖龍頭  37   
3.2.2.4 開啟座墊下置物箱  37   3.2.2.5 Gogoro Smart Coin  37     
3.2.3 ⼿機做為遙控器時  38   3.2.3.1 開啟系統電源並解鎖龍頭  38  
3.2.3.2 關閉系統電源及上鎖龍頭  38   
3.2.3.3 雙重防盜鎖  38   
3.2.3.4 開啟座墊下置物箱  39     
3.2.4 ⼿機做為免鑰匙智慧感應器時（智慧感應解鎖）  40  
3.2.4.1 開啟系統電源並解鎖龍頭  40   
3.2.4.2 關閉系統電源  40     
3.2.5 倒數計時⾃動上鎖  41     2          
3.2.6 快速關機組合鍵  41   
3.2.7 休眠模式  42     
4. 上路騎乘  43   
4.1 預估剩餘電量可⾏駛⾥程  43   
4.2', metadata={'location': {'s3Location': {'uri': 's3://gen-ai-hackton/z8ofwup116h2dnin1d253jbhk88t.pdf'}, 'type': 'S3'}, 'score': 0.5185076})]



'''