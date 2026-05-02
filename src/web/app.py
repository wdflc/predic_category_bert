
import uvicorn
from fastapi import FastAPI

from src.web.router import predict_router

# 创建FastAPI应用实例
app = FastAPI(title="商品标题分类API")

# 注册路由
app.include_router(predict_router)

# 启动应用
def run_app():
 uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000)
