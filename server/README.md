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
curl -X 'POST'   'http://127.0.0.1:8000/chat'  \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"}}, {"type": "text", "text": "please describe the image in detail"}]}]}'
```
