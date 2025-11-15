# Hướng Dẫn Chạy Pipeline (Đã Tối Ưu Hóa)

## 🚀 Chạy Pipeline Nhanh

### Bước 1: Kiểm tra cài đặt Memory Optimization

Mở file `src/config.py` và kiểm tra phần `MEMORY_OPTIMIZATION`:

```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': True,   # Đã bật để tránh lỗi bộ nhớ
    'sample_fraction': 0.1,   # Sử dụng 10% data
    ...
}
```

### Bước 2: Chạy Pipeline

```bash
# Windows PowerShell
$env:PYTHONIOENCODING='utf-8'
py -m src.pipelines._04_run_pipeline

# Hoặc Linux/Mac
export PYTHONIOENCODING=utf-8
python -m src.pipelines._04_run_pipeline
```

## ⚙️ Tùy Chỉnh Memory Optimization

### Nếu vẫn gặp lỗi bộ nhớ:

**Cách 1: Sử dụng script helper**
```bash
# Giảm xuống 5% data, giới hạn 10 products, 2 stores
py scripts/enable_memory_optimization.py --enable \
    --sample-fraction 0.05 \
    --max-products 10 \
    --max-stores 2 \
    --max-time 24
```

**Cách 2: Sửa trực tiếp trong config.py**
```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': True,
    'sample_fraction': 0.05,      # Giảm xuống 5%
    'max_products': 10,           # Chỉ 10 products
    'max_stores': 2,              # Chỉ 2 stores
    'max_time_periods': 24,       # Chỉ 24 hours
    'use_chunking': True,
    'chunk_size': 50000,          # Giảm chunk size
}
```

### Nếu máy có RAM cao (>16GB):

```bash
# Tắt sampling để dùng toàn bộ data
py scripts/enable_memory_optimization.py --disable
```

Hoặc sửa trong `src/config.py`:
```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': False,  # Tắt sampling
    ...
}
```

## 📊 Các Cấu Hình Đề Xuất

### Máy RAM thấp (< 8GB)
```python
'sample_fraction': 0.05,      # 5% data
'max_products': 5,
'max_stores': 1,
'max_time_periods': 24,
'chunk_size': 50000,
```

### Máy RAM trung bình (8-16GB)
```python
'sample_fraction': 0.1,       # 10% data (mặc định)
'max_products': None,         # Không giới hạn
'max_stores': None,
'max_time_periods': None,
'chunk_size': 100000,
```

### Máy RAM cao (> 16GB)
```python
'enable_sampling': False,     # Tắt sampling
'sample_fraction': 1.0,
'max_products': None,
'max_stores': None,
'max_time_periods': None,
'chunk_size': 200000,
```

## 🔍 Troubleshooting

### Lỗi: "Unable to allocate memory"
1. Giảm `sample_fraction` xuống 0.05 hoặc 0.01
2. Thêm giới hạn: `max_products=10`, `max_stores=2`
3. Giảm `chunk_size` xuống 50000

### Pipeline chạy quá chậm
1. Tăng `chunk_size` lên 200000
2. Tăng `sample_fraction` nếu có thể
3. Kiểm tra các process khác đang sử dụng CPU/RAM

### Muốn dùng toàn bộ data
1. Tắt sampling: `enable_sampling=False`
2. Set tất cả limits về `None`
3. Đảm bảo máy có đủ RAM (>16GB)

## 📚 Tài Liệu Thêm

- Chi tiết về Memory Optimization: `docs/MEMORY_OPTIMIZATION.md`
- Hướng dẫn đầy đủ: `docs/QUICKSTART.md`


