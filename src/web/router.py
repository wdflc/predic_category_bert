from fastapi import APIRouter, HTTPException

from src.web.schemas import PredictRequest, PredictResponse
from src.web.service import predict_service

# 创建路由
predict_router = APIRouter(tags=["预测接口"])

# 预测接口
@predict_router.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
 try:
  text = request.text.strip()
  if not text:
   raise HTTPException(status_code=400, detail="输入文本不能为空")

  pred_id, pred_label, confidence = predict_service(text)

  return PredictResponse(text=text, pred_id=pred_id, pred_label=pred_label, confidence=confidence)

 except Exception as e:
  raise HTTPException(status_code=500, detail=f"预测失败:{str(e)}")