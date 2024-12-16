## install dependency
1. install WePOINTS. see: [README](../README.md)
2. install ray:
    
```bash
pip install "ray[serve]"
```

## start server

```bash
serve run ray_server:build_app
```

## invoke server
```bash
# send image by url:
python ./demo_client.py --url="https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"

# or send image by base64
wget -O "test_image.jpeg" "https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"
python ./demo_client.py --file="./test_image.jpeg"
```