# convert_latex_to_docx.py
from tempfile import NamedTemporaryFile
import subprocess
import os
from fastapi.responses import FileResponse

async def convert_latex_to_docx(latex_text: str = None, file: bytes = None):
    # Tạo file tạm thời để lưu LaTeX
    with NamedTemporaryFile(delete=False, suffix=".tex") as temp_tex:
        if latex_text:
            temp_tex.write(latex_text.encode())
        elif file:
            temp_tex.write(file)
        temp_tex_path = temp_tex.name

    # Đường dẫn output cho file .docx
    output_docx_path = temp_tex_path.replace(".tex", ".docx")

    try:
        # Sử dụng Pandoc để chuyển đổi sang .docx
        subprocess.run(["pandoc", temp_tex_path, "-o", output_docx_path], check=True)
    except subprocess.CalledProcessError:
        raise Exception("Failed to convert LaTeX to DOCX")

    # Trả về file .docx để tải xuống
    return FileResponse(output_docx_path, filename="output.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
