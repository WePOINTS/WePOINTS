import base64
import requests
import json
import argparse
from typing import Optional


def create_request(url: Optional[str] = None,
                   file: Optional[str] = None) -> dict:
    if url is not None:
        payload_url = url
    else:
        with open(file, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        payload_url = f"data:image/jpeg;base64,{image_base64}"
    return {
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": payload_url,
                }
            }, {
                "type": "text",
                "text": "please describe the image in detail"
            }]
        }]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    assert (args.url is not None or args.file
            is not None) and not (args.url is None and args.file is None)

    req = create_request(args.url, args.file)
    response = requests.post(f"{args.addr}/run", data=json.dumps(req))
    print(response.json())


if __name__ == "__main__":
    main()
