
from pydantic import BaseModel

# 预测请求体类型
class PredictRequest(BaseModel):
 text: str

# 预测响应体类型
class PredictResponse(BaseModel):
 text: str
 pred_id: int
 pred_label: str