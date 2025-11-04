import sys
from pathlib import Path

# Xác định đường dẫn GỐC của thư mục /src
SRC_ROOT = Path(__file__).parent.resolve()

# Thêm đường dẫn /src vào sys.path nếu nó chưa có
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
    # print(f"Đã thêm {SRC_ROOT} vào sys.path") # <-- Tạm thời comment lại
    # Hoặc sửa thành tiếng Anh:
    print(f"Added {SRC_ROOT} to sys.path")