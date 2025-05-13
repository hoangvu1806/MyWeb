from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os, json, ast
import sys
# Thêm thư mục chứa "decision-tree" vào sys.path
sys.path.append(os.path.abspath("E:/MyWeb"))
# Import module bằng __import__
built_tree = __import__("decision-tree.built_tree", fromlist=["built_tree"])
#============================================================================

class Message(BaseModel):
    name: str
    email: str
    form_message: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hoangvu.id.vn", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Gắn các thư mục static (CSS, JS, Images) cho thư mục "home"
app.mount("/css", StaticFiles(directory="home/css"), name="css")
app.mount("/js", StaticFiles(directory="home/js"), name="js")
app.mount("/images", StaticFiles(directory="home/images"), name="images")
app.mount("/vendor", StaticFiles(directory="home/vendor"), name="vendor")
app.mount("/Error", StaticFiles(directory="home/Error"), name="error")
app.mount("/ajax", StaticFiles(directory="home/ajax"), name="ajax")
app.mount("/decision-tree/templates/css",StaticFiles(directory="decision-tree/templates/css"),name="css_dt")
app.mount("/decision-tree/templates/js",StaticFiles(directory="decision-tree/templates/js"),name="js_dt")
app.mount("/decision-tree/templates/img",StaticFiles(directory="decision-tree/templates/img"),name="img_dt")
app.mount("/linear-regression/css",StaticFiles(directory="linear-regression/templates/css"),name="css_lr")
app.mount("/chatting/templates",StaticFiles(directory="chatting/templates"),name="templates_chat")
app.mount("/tex2docx/templates",StaticFiles(directory="tex2docx/templates"),name="templates_tex2docx")
app.mount("/ezclip/resources",StaticFiles(directory="ezclip/resources"),name="ezclip_resources")

# Cấu hình Jinja2 templates cho từng thư mục
templates_home = Jinja2Templates(directory="home")
templates_errors = Jinja2Templates(directory="home/Error")
templates_dt = Jinja2Templates(directory="decision-tree/templates")
templates_lr = Jinja2Templates(directory="linear-regression/templates")
templates_chat = Jinja2Templates(directory="chatting/templates")
templates_tex2docx = Jinja2Templates(directory="tex2docx/templates")
templates_ezclip = Jinja2Templates(directory="ezclip")

#============================================================================

# Điều hướng mặc định từ trang gốc "/" đến "/home"
@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/home")

# Endpoint cho trang chủ (mặc định vào /home)
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates_home.TemplateResponse("index.html", {"request": request})

# Xử lý HTTPException (ví dụ: 404 lỗi không tìm thấy)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates_errors.TemplateResponse("404.html", {"request": request}, status_code=404)
    return HTMLResponse(content=f"Error: {exc.detail}", status_code=exc.status_code)

# Xử lý ValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return HTMLResponse(content=f"Validation Error: {exc.errors()}", status_code=422)

# Endpoint cho trang Decision Tree
@app.get("/decision-tree", response_class=HTMLResponse)
async def decision_tree_home(request: Request):
    return templates_dt.TemplateResponse("index.html", {"request": request})

@app.post("/decision-tree/build-tree")
async def build_tree(csvFile: UploadFile = File(...),targetColumn: str = Form(...), dropColumn: str = Form(None),maxDepth: int = Form(...),minSplit: int = Form(...),criterion: str = Form(None)):
    try:
        dropColumns = ast.literal_eval(dropColumn) if dropColumn else []
        print(dropColumns)
        tree, features, metric, split_ratio, seed = built_tree.main(csvFile.file, targetColumn, dropColumns, maxDepth, minSplit, criterion)
        print(tree)
        return {"status": "Build Tree successful", "tree": tree, "features": features, "metric": metric, "split_ratio": split_ratio, "seed": seed}
    except Exception as e:
        return {"status": "Build Tree failed", "error": str(e)}
    
# Endpoint cho trang Linear Regression
@app.get("/linear-regression", response_class=HTMLResponse)
async def linear_regression_home(request: Request):
    return templates_lr.TemplateResponse("index.html", {"request": request})

@app.get("/chatting", response_class=HTMLResponse)
async def chatting(request: Request):
    return templates_chat.TemplateResponse("index.html", {"request": request})

# Endpoint tải CV
@app.get("/home/CV.pdf", response_class=FileResponse)
async def download_cv():
    file_path = r"D:\MyWeb\home\CV.pdf"
    return FileResponse(file_path, filename="CV.pdf", media_type="application/pdf")

@app.post("/api/send-message")
async def send_message(message: Message, request: Request):
    data = {
        "name": message.name,
        "email": message.email,
        "form_message": message.form_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip_address": request.client.host,
        "user_agent": request.headers.get("User-Agent"),
    }
    file_path = os.path.join("home", "viewer", "message.json")
    try:
        # Đọc dữ liệu cũ từ file nếu có
        with open(file_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        messages = []
    # Thêm tin nhắn mới vào danh sách
    messages.append(data)
    # Ghi lại vào file JSON với ensure_ascii=False
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    return {"status": "success", "message": "Message received"}


@app.get("/tex2docx", response_class=HTMLResponse)
async def tex2docx(request: Request):
    return templates_tex2docx.TemplateResponse("index.html", {"request": request})


@app.post("/tex2docx/convert")
async def latex_to_docx(latex_text: str = Form(None), file: UploadFile = File(None)):
    from tex2docx.tex2docx import convert_latex_to_docx
    if latex_text:
        return await convert_latex_to_docx(latex_text=latex_text)
    elif file:
        file_content = await file.read()
        return await convert_latex_to_docx(file=file_content)
    else:
        return {"error": "Please provide either LaTeX text or a file to convert."}

@app.get("/ezclip")
async def ezclip(request: Request):
    return templates_ezclip.TemplateResponse("index.html", {"request": request})

# Endpoint để tải file theo nền tảng
@app.get("/ezclip/downloads", response_class=FileResponse)
async def download_file(platform: str = Query(..., description="Platform: windows, linux, mac")):
    # Đường dẫn thư mục chứa các file tải
    DOWNLOADS_DIR = r"D:\MyWeb\ezclip\resources\setup"
    # Bản đồ ánh xạ từ nền tảng sang tên file
    file_map = {
        "windows": "EzClip Setup 1.1.2.exe",
        "linux": "EzClip Setup 1.1.2.AppImage",
        "mac": "EzClip Setup 1.1.2.dmg",
    }
    if platform.lower() not in file_map:
        raise HTTPException(status_code=400, detail="Invalid platform. Choose from: windows, linux, mac")
    file_name = file_map[platform.lower()]
    file_path = os.path.join(DOWNLOADS_DIR, file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found for the specified platform")
    # Trả về file tương ứng
    return FileResponse(file_path, filename=file_name, media_type="application/octet-stream")

@app.post("/ezclip/send-message")
async def send_message(message: Message, request: Request):
    data = {
        "name": message.name,
        "email": message.email,
        "form_message": message.form_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip_address": request.client.host,
        "user_agent": request.headers.get("User-Agent"),
    }
    file_path = os.path.join("ezclip", "feedback.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Đọc dữ liệu cũ từ file nếu có
        with open(file_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        messages = []

    # Thêm tin nhắn mới vào danh sách
    messages.append(data)

    # Ghi lại vào file JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    return {"status": "success", "message": "Message received"}


# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
