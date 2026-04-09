import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# 1. 完整的花卉名称字典（为了演示完整性保留）
idx_to_class = {
    0: '粉色月见草', 1: '硬叶兜兰', 2: '风铃草', 3: '香豌豆', 4: '金盏花', 
    5: '卷丹百合', 6: '蝴蝶兰', 7: '鹤望兰', 8: '乌头', 9: '蓝刺头', 10: '金鱼草', 
    11: '款冬', 12: '帝王花', 13: '翼蓟', 14: '黄鸢尾', 15: '金莲花', 16: '紫松果菊', 
    17: '秘鲁百合', 18: '桔梗', 19: '巨型白马蹄莲', 20: '火百合', 21: '针垫花', 22: '贝母', 
    23: '红姜花', 24: '葡萄风信子', 25: '虞美人', 26: '千日红', 27: '无茎龙胆', 28: '洋蓟', 
    29: '须苞石竹', 30: '康乃馨', 31: '宿根福禄考', 32: '黑种草', 33: '波斯菊', 34: '高山刺芹', 
    35: '卡特兰', 36: '蓝眼菊', 37: '大星芹', 38: '姜荷花', 39: '圣诞玫瑰', 40: '非洲菊', 
    41: '黄水仙', 42: '唐菖蒲', 43: '一品红', 44: '洋桔梗', 45: '桂竹香', 46: '万寿菊', 
    47: '毛茛', 48: '牛眼菊', 49: '蒲公英', 50: '矮牵牛', 51: '三色堇', 52: '报春花', 
    53: '向日葵', 54: '天竺葵', 55: '大丽花', 56: '山桃草', 57: '老鹳草', 58: '橘色大丽花', 
    59: '粉黄大丽花', 60: '黄姜花', 61: '秋明菊', 62: '黑心菊', 63: '银旋花', 64: '加州罂粟', 
    65: '万维菊', 66: '春番红花', 67: '鸢尾', 68: '秋牡丹', 69: '树罂粟', 70: '勋章菊', 
    71: '杜鹃花', 72: '睡莲', 73: '玫瑰', 74: '曼陀罗', 75: '牵牛花', 76: '西番莲', 
    77: '莲花', 78: '蟾蜍百合', 79: '红掌', 80: '鸡蛋花', 81: '铁线莲', 82: '木槿', 
    83: '耧斗菜', 84: '沙漠玫瑰', 85: '树锦葵', 86: '玉兰', 87: '仙客来', 88: '西洋菜', 
    89: '美人蕉', 90: '朱顶红', 91: '蜂香薄荷', 92: '空气凤梨', 93: '毛地黄', 94: '三角梅', 
    95: '茶花', 96: '锦葵', 97: '翠芦莉', 98: '凤梨花', 99: '天人菊', 100: '凌霄花', 101: '射干'
}

# 2. 启动时将 ONNX 模型加载到内存（只需加载一次）
print("正在启动 ONNX 推理引擎...")
ort_session = ort.InferenceSession("resnet_flowers.onnx")
# 获取模型的输入节点名称（我们在转换时起名叫 'input'）
input_name = ort_session.get_inputs()[0].name

def preprocess_image_numpy(image_bytes):
    """
    完全使用 NumPy 手写图像预处理，脱离 PyTorch 依赖。
    逻辑等同于：Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 1. Resize (短边缩放至256) 和 CenterCrop (224x224)
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    image = image.crop((left, top, left + 224, top + 224))
    
    # 2. 转为 NumPy 数组并归一化到 [0, 1] (等同于 ToTensor)
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 3. 标准化 Normalize (ImageNet 的 mean 和 std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # 4. 调整通道顺序: 从 (Height, Width, Channels) 变为 (Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # 5. 增加 Batch 维度: 变为 (1, Channels, Height, Width)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 执行轻量级预处理
    input_data = preprocess_image_numpy(contents)
    
    # 🚀 执行极速推理！
    # ort_session.run 需要的参数：(输出节点名称列表(填None即可), 输入数据字典)
    outputs = ort_session.run(None, {input_name: input_data})
    
    # 解析输出 (outputs[0] 是一个 shape 为 [1, 102] 的数组)
    logits = outputs[0]
    class_id = int(np.argmax(logits, axis=1)[0]) # 找到概率最大的索引
    
    flower_name = idx_to_class.get(class_id, "未知花卉")
        
    return {
        "class_id": class_id, 
        "flower_name": flower_name,
        "engine": "ONNX Runtime", # 加个标记证明我们在用高级引擎
        "message": "Prediction successful"
    }

from fastapi.responses import HTMLResponse

@app.get("/")
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>花卉识别AI</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin-top: 50px; }
            .container { border: 1px solid #ddd; padding: 20px; border-radius: 10px; display: inline-block; }
            input[type="file"] { margin: 20px 0; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; font-size: 16px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🌸 AI 花卉识别系统</h2>
            <p>基于 ResNet18 + ONNX 极速推理引擎</p>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required><br>
                <button type="submit">立即预测</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

#uvicorn main_ONNX:app --reload
#http://127.0.0.1:8000/docs